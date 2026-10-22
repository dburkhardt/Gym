# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
import math
import random
import time
from contextlib import asynccontextmanager, nullcontext
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Literal, Mapping, Optional, TypeVar

from aiohttp import ClientResponseError
from fastapi import HTTPException
from pydantic import BaseModel, Field, model_validator

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    MAX_NUM_TRIES,
    NeMoGymAsyncOpenAI,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymCompletion,
    NeMoGymCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    PermanentEndpointError,
    ReasoningEffort,
)


LOG = logging.getLogger(__name__)
ResponseT = TypeVar("ResponseT")


class UpstreamRetriesExhaustedError(RuntimeError):
    """An upstream retry policy ran out of attempts.

    Keeps the final provider error's HTTP status and body, which model-call capture reads from
    the raised exception; the error still reaches callers as HTTP 500.
    """

    def __init__(self, attempts: int, last_error: Optional[Exception]) -> None:
        super().__init__(f"Upstream provider request failed after {attempts} attempts")
        self.status = getattr(last_error, "status", None)
        self.response_content = getattr(last_error, "response_content", None)


class UpstreamAdmissionTimeoutError(RuntimeError):
    """Queued admission expired before the next provider attempt was started.

    Earlier attempts, if any, remain in the capture ledger; this does not assert
    that the entire logical request had zero physical sends.
    """


class _AdmissionBudget:
    """Per-logical-call cumulative waiting time, never provider execution time."""

    def __init__(self, seconds: float | None):
        self.remaining = seconds

    @asynccontextmanager
    async def waiting(self):
        if self.remaining is None:
            yield
            return
        if self.remaining <= 0:
            raise UpstreamAdmissionTimeoutError("Upstream queued admission budget exhausted before next attempt")
        start = asyncio.get_running_loop().time()
        timer = asyncio.timeout(self.remaining)
        try:
            async with timer:
                yield
        except TimeoutError as error:
            if timer.expired():
                raise UpstreamAdmissionTimeoutError(
                    "Upstream queued admission budget exhausted before next attempt"
                ) from error
            raise
        finally:
            self.remaining = max(0.0, self.remaining - (asyncio.get_running_loop().time() - start))

    async def sleep(self, seconds: float):
        async with self.waiting():
            await asyncio.sleep(seconds)


class UpstreamRetryPolicy(BaseModel):
    """Retry policy for one logical request to the upstream provider."""

    max_attempts: int = Field(
        default=1,
        ge=1,
        description="Total provider attempts, including the first attempt.",
    )
    pre_request_jitter_seconds: Optional[tuple[float, float]] = Field(
        default=None,
        description=("Optional uniform sleep range sampled before every provider attempt, including the first."),
    )
    backoff_initial_seconds: float = Field(
        default=1.0,
        ge=0,
        description="Base delay after the first failed attempt.",
    )
    backoff_multiplier: float = Field(
        default=2.0,
        gt=0,
        description="Multiplier applied to the base delay after each failure.",
    )
    backoff_jitter_fraction: float = Field(
        default=0.0,
        ge=0,
        description=("Uniform additive jitter from zero through this fraction of the current base delay."),
    )
    terminal_http_status_codes: frozenset[int] = Field(
        default_factory=frozenset,
        description="Upstream HTTP statuses that must fail without retrying.",
    )

    @model_validator(mode="after")
    def validate_pre_request_jitter(self) -> "UpstreamRetryPolicy":
        if self.pre_request_jitter_seconds is None:
            return self

        lower, upper = self.pre_request_jitter_seconds
        if lower < 0 or upper < lower:
            raise ValueError("pre_request_jitter_seconds must be a nonnegative, ordered two-value range")
        return self


class SimpleModelServerConfig(BaseResponsesAPIModelConfig):
    openai_base_url: str
    openai_api_key: str
    openai_model: str

    extra_body: Dict[str, Any] = Field(default_factory=dict)
    raw_completions_enabled: bool = False

    @model_validator(mode="after")
    def validate_raw_completion_capability(self):
        if self.raw_completions_enabled:
            NeMoGymCompletionCreateParamsNonStreaming.model_validate(
                {"model": self.openai_model, "prompt": "", "max_tokens": 1} | self.extra_body
            )
        return self

    openai_default_headers: Dict[str, str] = Field(default_factory=dict)
    max_http_attempts: int = Field(default=MAX_NUM_TRIES, ge=1)
    upstream_rate_limit_cooldown_seconds: Optional[float] = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Opt-in shared 429 cooldown fallback. Honors Retry-After; retries only explicit 429 responses.",
    )

    upstream_rate_limit_shared_directory: Optional[Path] = None
    upstream_rate_limit_shared_namespace: Optional[str] = Field(default=None, min_length=1, max_length=128)

    reasoning_effort_none_replacement: Optional[ReasoningEffort] = Field(
        default=None,
        description=(
            "Opt-in stand-in for a response's reasoning.effort 'none', for callers "
            "that cannot accept 'none' (some providers, e.g. NVIDIA's "
            "Responses-compatible endpoint, return it). Null, the default, reports "
            "the provider's value unchanged."
        ),
    )

    max_concurrent_requests: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Cap on in-flight upstream requests from this server (per-process "
            "asyncio.Semaphore). Set on rate-limited endpoints (e.g. Gemini) "
            "to stay under quota; None = unlimited."
        ),
    )
    upstream_max_num_tries: Optional[Literal[1]] = Field(
        default=None,
        description=(
            "Set to 1 so each provider call makes exactly one HTTP attempt. This "
            "disables the outbound client's transport and HTTP-status retry layers, "
            "replacing their generic-error and max_http_attempts limits. Required "
            "by an upstream_retry_policy with multiple attempts. None preserves "
            "existing behavior."
        ),
    )
    upstream_request_timeout_seconds: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Total time limit for each provider attempt, not a deadline for the "
            "whole retry schedule. None means no limit."
        ),
    )
    upstream_admission_timeout_seconds: Optional[float] = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Cumulative queued time per logical call: jitter, cooldown, semaphore acquisition "
            "and retry backoff. Provider execution is excluded and retains its separate "
            "per-attempt timeout. None preserves unbounded queued admission."
        ),
    )
    upstream_connect_timeout_seconds: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Time limit for each provider attempt to obtain a connection, including "
            "waiting for a free aiohttp connection-pool slot. Requires "
            "upstream_request_timeout_seconds."
        ),
    )
    upstream_pool_timeout_seconds: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Time limit for each attempt to wait for one of this server's "
            "max_concurrent_requests slots, like a provider client's pool-acquire "
            "timeout. It does not time the provider call itself."
        ),
    )
    upstream_exhausted_token_cooldown_seconds: Optional[float] = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Opt-in: trust x-ratelimit-remaining-tokens <= 0 on successful provider responses "
            "and delay future admission by this explicit duration. No inferred quota/reset or resend. "
            "Requires upstream_rate_limit_cooldown_seconds. Disabled by default."
        ),
    )
    propagate_upstream_http_status_codes: frozenset[int] = Field(
        default_factory=frozenset,
        description=(
            "Upstream HTTP error statuses returned to the caller with that status "
            "and the provider's error body, instead of HTTP 500. A status that "
            "upstream_retry_policy retries until it is exhausted still returns "
            "HTTP 500. Empty keeps the historical generic exception behavior."
        ),
    )
    upstream_retry_policy: UpstreamRetryPolicy = Field(
        default_factory=UpstreamRetryPolicy,
        description=(
            "Optional model-server retry layer around provider calls. The "
            "default policy makes one call with no added delay. Each retry "
            "attempt reacquires the upstream concurrency slot."
        ),
    )

    @model_validator(mode="after")
    def validate_upstream_controls(self) -> "SimpleModelServerConfig":
        if (self.upstream_rate_limit_shared_directory is None) != (self.upstream_rate_limit_shared_namespace is None):
            raise ValueError("Shared cooldown requires both directory and namespace")
        if (
            self.upstream_exhausted_token_cooldown_seconds is not None
            and self.upstream_rate_limit_cooldown_seconds is None
        ):
            raise ValueError("Exhausted-token admission requires upstream_rate_limit_cooldown_seconds")
        if self.upstream_rate_limit_shared_directory is not None and self.upstream_rate_limit_cooldown_seconds is None:
            raise ValueError("Shared cooldown requires upstream_rate_limit_cooldown_seconds")
        if self.upstream_rate_limit_cooldown_seconds is not None and self.upstream_max_num_tries != 1:
            raise ValueError("Rate-limit cooldown requires upstream_max_num_tries=1")
        if self.upstream_retry_policy.max_attempts > 1 and self.upstream_max_num_tries != 1:
            raise ValueError(
                "An upstream retry policy with multiple attempts requires "
                "upstream_max_num_tries=1 so no inner retry layer can add "
                "provider attempts"
            )
        if self.upstream_pool_timeout_seconds is not None and self.max_concurrent_requests is None:
            raise ValueError("upstream_pool_timeout_seconds requires max_concurrent_requests")
        if self.upstream_connect_timeout_seconds is not None and self.upstream_request_timeout_seconds is None:
            raise ValueError("upstream_connect_timeout_seconds requires upstream_request_timeout_seconds")
        invalid_status_codes = sorted(
            status for status in self.propagate_upstream_http_status_codes if status < 400 or status > 599
        )
        if invalid_status_codes:
            raise ValueError(
                "propagate_upstream_http_status_codes must contain HTTP error "
                f"statuses in [400, 599], received {invalid_status_codes}"
            )
        return self

    drop_input_reasoning_items: bool = Field(
        default=False,
        description=(
            "Strip type=reasoning items from the Responses API input before the "
            "upstream call. Workaround for endpoints (e.g. NVIDIA-hosted gpt-oss) "
            "that 500 with KeyError 'content' on their own content-less reasoning "
            "items when echoed back across tool-use turns."
        ),
    )


class SimpleModelServer(SimpleResponsesAPIModel):
    config: SimpleModelServerConfig

    def _should_propagate_upstream_http_error(self, error: ClientResponseError) -> bool:
        # OpenAI and subclasses such as LiteLLM retain their opt-in HTTP policy.
        return error.status in self.config.propagate_upstream_http_status_codes

    def model_post_init(self, context):
        self._rate_limit_until = 0.0
        self._shared_cooldown = None
        if self.config.upstream_rate_limit_shared_directory is not None:
            from responses_api_models.openai_model.shared_cooldown import SharedCooldown

            self._shared_cooldown = SharedCooldown(
                self.config.upstream_rate_limit_shared_directory,
                self.config.upstream_rate_limit_shared_namespace,
                self.config.openai_base_url,
                self.config.openai_model,
            )
            self._shared_cooldown.remaining()  # Reject mismatched/corrupt scope before dispatch.

        self._client = NeMoGymAsyncOpenAI(
            base_url=self.config.openai_base_url,
            api_key=self.config.openai_api_key,
            default_headers=self.config.openai_default_headers,
            max_num_tries=self.config.upstream_max_num_tries,
            request_timeout_seconds=self.config.upstream_request_timeout_seconds,
            connect_timeout_seconds=self.config.upstream_connect_timeout_seconds,
            max_http_attempts=self.config.max_http_attempts,
        )
        if self.config.upstream_exhausted_token_cooldown_seconds is not None:
            self._client.set_response_headers_observer(observer=self._observe_quota_headers)
        self._semaphore = (
            asyncio.Semaphore(self.config.max_concurrent_requests)
            if self.config.max_concurrent_requests is not None
            else nullcontext()
        )

        return super().model_post_init(context)

    @asynccontextmanager
    async def _upstream_request_slot(self):
        """Hold one of this server's provider slots for a single attempt.

        ``upstream_pool_timeout_seconds`` bounds only the wait for a free slot,
        like a provider client's pool-acquire timeout; the provider call itself
        is bounded per attempt by ``upstream_request_timeout_seconds``. An
        acquired slot is always released here.
        """

        if self.config.max_concurrent_requests is None or self.config.upstream_pool_timeout_seconds is None:
            async with self._semaphore:
                yield
            return

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.config.upstream_pool_timeout_seconds,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                "Timed out acquiring an upstream provider slot after "
                f"{self.config.upstream_pool_timeout_seconds:g} seconds"
            ) from exc

        try:
            yield
        finally:
            self._semaphore.release()

    @asynccontextmanager
    async def _admitted_upstream_slot(self, budget: _AdmissionBudget):
        slot = self._upstream_request_slot()
        async with budget.waiting():
            await slot.__aenter__()
        try:
            yield
        finally:
            await slot.__aexit__(None, None, None)

    async def _shared_cooldown_call(self, operation: Callable, *args):
        """Settle bounded local state I/O before returning caller cancellation."""
        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as cancellation:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            try:
                task.result()
            except Exception as error:
                cancellation.add_note(f"Shared cooldown state settlement failed: {type(error).__name__}")
            raise cancellation

    async def _cooldown_remaining(self) -> float:
        local = max(0.0, self._rate_limit_until - time.monotonic())
        if self._shared_cooldown is None:
            return local
        return max(local, await self._shared_cooldown_call(self._shared_cooldown.remaining))

    async def _observe_quota_headers(self, status: int, headers: Mapping[str, str]) -> None:
        """Publish observed pressure for future calls, never retry an accepted call."""
        delay = self.config.upstream_exhausted_token_cooldown_seconds
        if delay is None or not 200 <= status < 300:
            return
        value = next((v for k, v in headers.items() if k.lower() == "x-ratelimit-remaining-tokens"), None)
        try:
            remaining = float(value)
        except (TypeError, ValueError, OverflowError):
            return
        if not math.isfinite(remaining) or remaining > 0:
            return
        self._rate_limit_until = max(self._rate_limit_until, time.monotonic() + delay)
        if self._shared_cooldown is not None:
            await self._shared_cooldown_call(self._shared_cooldown.extend, delay)
        LOG.info("Provider token-pressure cooldown: remaining=%s delay_seconds=%s", remaining, delay)

    async def _call_with_cooldown(
        self, operation: Callable[[], Awaitable[ResponseT]], budget: _AdmissionBudget
    ) -> ResponseT:
        """Wait outside the slot; only observed 429 rejection is safe to retry."""
        policy = self.config.upstream_retry_policy
        for attempt in range(policy.max_attempts):
            if policy.pre_request_jitter_seconds is not None:
                await budget.sleep(random.uniform(*policy.pre_request_jitter_seconds))
            while True:
                async with budget.waiting():
                    delay = await self._cooldown_remaining()
                if delay > 0:
                    await budget.sleep(delay)
                async with self._admitted_upstream_slot(budget):
                    # Another request may have established a cooldown while we queued.
                    async with budget.waiting():
                        still_cooling = await self._cooldown_remaining() > 0
                    if still_cooling:
                        continue
                    try:
                        return await operation()
                    except PermanentEndpointError:
                        raise
                    except ClientResponseError as error:
                        if error.status != 429:
                            raise
                        delay = self.config.upstream_rate_limit_cooldown_seconds
                        value = (error.headers or {}).get("Retry-After")
                        try:
                            parsed = float(value)
                        except (TypeError, ValueError):
                            try:
                                parsed = parsedate_to_datetime(value).timestamp() - time.time()
                            except (TypeError, ValueError, OverflowError, AttributeError):
                                parsed = float("nan")
                        if math.isfinite(parsed) and parsed >= 0:
                            delay = parsed
                        self._rate_limit_until = max(self._rate_limit_until, time.monotonic() + delay)
                        if self._shared_cooldown is not None:
                            await self._shared_cooldown_call(self._shared_cooldown.extend, delay)
                        if error.status in policy.terminal_http_status_codes or attempt + 1 == policy.max_attempts:
                            raise
                break
            # Backoff complements, rather than replaces, the shared provider wait.
            backoff = policy.backoff_initial_seconds * policy.backoff_multiplier**attempt
            await budget.sleep(backoff + random.uniform(0, backoff * policy.backoff_jitter_fraction))
        raise AssertionError("Validated retry policy always contains an attempt")

    async def _call_upstream(
        self,
        operation: Callable[[], Awaitable[ResponseT]],
    ) -> ResponseT:
        budget = _AdmissionBudget(self.config.upstream_admission_timeout_seconds)
        if self.config.upstream_rate_limit_cooldown_seconds is not None:
            return await self._call_with_cooldown(operation, budget)
        retry_policy = self.config.upstream_retry_policy
        if retry_policy.max_attempts == 1:
            # Preserve the ordinary model server's historical behavior: one
            # request owns its concurrency slot until its one client call has
            # returned (including any retries performed inside that client).
            if retry_policy.pre_request_jitter_seconds is not None:
                await budget.sleep(random.uniform(*retry_policy.pre_request_jitter_seconds))
            async with self._admitted_upstream_slot(budget):
                return await operation()

        last_error: Exception | None = None
        for attempt in range(retry_policy.max_attempts):
            if retry_policy.pre_request_jitter_seconds is not None:
                await budget.sleep(random.uniform(*retry_policy.pre_request_jitter_seconds))
            try:
                # Acquire per attempt so a failure releases its provider slot
                # before retry backoff instead of monopolizing capacity for
                # the full retry loop.
                async with self._admitted_upstream_slot(budget):
                    return await operation()
            except (PermanentEndpointError, UpstreamAdmissionTimeoutError):
                # The client has stopped calling a spent or unauthorized endpoint,
                # so every later attempt would fail without reaching the provider.
                raise
            except ClientResponseError as exc:
                if exc.status in retry_policy.terminal_http_status_codes:
                    raise
                last_error = exc
            except Exception as exc:
                last_error = exc

            if attempt + 1 >= retry_policy.max_attempts:
                break

            base_backoff = retry_policy.backoff_initial_seconds * retry_policy.backoff_multiplier**attempt
            retry_wait = base_backoff + random.uniform(
                0.0,
                base_backoff * retry_policy.backoff_jitter_fraction,
            )
            LOG.warning(
                "Upstream provider request failed (%d/%d); retrying after %.1f seconds: %s",
                attempt + 1,
                retry_policy.max_attempts,
                retry_wait,
                last_error,
            )
            await budget.sleep(retry_wait)

        raise UpstreamRetriesExhaustedError(retry_policy.max_attempts, last_error) from last_error

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model
        if self.config.drop_input_reasoning_items:
            input_items = body_dict.get("input")
            if isinstance(input_items, list):
                body_dict["input"] = [
                    item for item in input_items if not (isinstance(item, dict) and item.get("type") == "reasoning")
                ]

        async def create_and_validate() -> NeMoGymResponse:
            response_dict = await self._client.create_response(**body_dict)
            reasoning = response_dict.get("reasoning") if isinstance(response_dict, dict) else None
            if (
                self.config.reasoning_effort_none_replacement is not None
                and isinstance(reasoning, dict)
                and reasoning.get("effort") == "none"
            ):
                response_dict = dict(response_dict)
                response_dict["reasoning"] = {
                    **reasoning,
                    "effort": self.config.reasoning_effort_none_replacement,
                }
            return NeMoGymResponse.model_validate(response_dict)

        return await self._call_upstream(create_and_validate)

    async def completions(self, body: NeMoGymCompletionCreateParamsNonStreaming) -> NeMoGymCompletion:
        if not self.config.raw_completions_enabled:
            raise HTTPException(status_code=501, detail="Enable raw_completions_enabled after provider qualification")
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model
        # Configured extras must obey the same strict capability contract.
        body_dict = NeMoGymCompletionCreateParamsNonStreaming.model_validate(body_dict).model_dump(exclude_unset=True)

        async def create_and_validate() -> NeMoGymCompletion:
            response = NeMoGymCompletion.model_validate(await self._client.create_completion(**body_dict))
            if response.model != self.config.openai_model:
                raise ValueError("Raw completion response model does not match configured model")
            return response

        return await self._call_upstream(create_and_validate)

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model

        async def create_and_validate() -> NeMoGymChatCompletion:
            response_dict = await self._client.create_chat_completion(**body_dict)
            return NeMoGymChatCompletion.model_validate(response_dict)

        return await self._call_upstream(create_and_validate)


if __name__ == "__main__":
    SimpleModelServer.run_webserver()
