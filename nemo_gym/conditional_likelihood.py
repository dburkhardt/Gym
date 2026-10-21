# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline raw causal-likelihood contract; no endpoint is admitted by this module.

Mirrors lm-eval 0.4.2's causal token split/truncation where the context remains
an exact prefix. Boundary-merging tokenizers fail closed rather than silently
score a different string. Chat prompt NLL is not an implementation of this API.
"""

import math
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from nemo_gym.provider_dispatch import configuration_digest


class LikelihoodUnavailable(ValueError):
    """Exact raw continuation scoring cannot be established."""


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LikelihoodIdentity(_Contract):
    model: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    tokenizer: str = Field(min_length=1)
    tokenizer_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    tokenizer_config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    # Bind BOS/EOT/add-special-token policy as part of tokenizer configuration.
    prefix_token_id: StrictInt = Field(ge=0)
    conditioning: Literal["raw_causal_no_chat_template"] = "raw_causal_no_chat_template"


class LikelihoodRequest(_Contract):
    protocol: Literal["raw-causal-lm-eval-0.4.2-strict-boundary-v1"] = "raw-causal-lm-eval-0.4.2-strict-boundary-v1"
    context: str
    continuation: str = Field(min_length=1)
    identity: LikelihoodIdentity
    max_length: int = Field(gt=0)
    truncation: Literal["reject", "left_context_lm_eval_0.4.2"] = "reject"


class LikelihoodPlan(_Contract):
    request_digest: str
    identity: LikelihoodIdentity
    context_token_ids: tuple[StrictInt, ...]
    continuation_token_ids: tuple[StrictInt, ...]
    # Actual causal input excludes final target. Positions index its predicted logits.
    model_input_token_ids: tuple[StrictInt, ...]
    continuation_logit_positions: tuple[StrictInt, ...]
    left_context_tokens_dropped: int
    trailing_context_characters_moved: int


class LikelihoodObservation(_Contract):
    request_digest: str
    identity: LikelihoodIdentity
    model_input_token_ids: tuple[StrictInt, ...]
    continuation_token_ids: tuple[StrictInt, ...]
    continuation_logit_positions: tuple[StrictInt, ...]
    token_logprobs: tuple[float, ...]
    summed_logprob: float


def prepare_likelihood(request: LikelihoodRequest, *, encode: Callable[[str], list[int]]) -> LikelihoodPlan:
    """Build auditable boundaries using the backend's separately qualified tokenizer."""
    context, continuation = request.context, request.continuation
    moved = len(context) - len(context.rstrip())
    if not context:
        context_ids, continuation_ids = [request.identity.prefix_token_id], encode(continuation)
    else:
        if moved:
            continuation, context = context[-moved:] + continuation, context[:-moved]
        context_ids, whole = encode(context), encode(context + continuation)
        # Upstream splits by token count without checking prefix equality. Reject
        # that ambiguous case; independently encoding continuation is also wrong.
        if whole[: len(context_ids)] != context_ids:
            raise LikelihoodUnavailable(
                "Tokenizer merges across context boundary; exact continuation alignment unavailable"
            )
        continuation_ids = whole[len(context_ids) :]
    if not context_ids or not continuation_ids:
        raise LikelihoodUnavailable("Empty tokenized context or continuation")
    if any(type(t) is not int or t < 0 for t in context_ids + continuation_ids):
        raise LikelihoodUnavailable("Invalid tokenizer IDs")
    if len(continuation_ids) > request.max_length:
        raise LikelihoodUnavailable("Continuation exceeds maximum length; continuation truncation forbidden")
    dropped = max(0, len(context_ids) + len(continuation_ids) - request.max_length - 1)
    if dropped and request.truncation == "reject":
        raise LikelihoodUnavailable("Request requires context truncation")
    causal = (context_ids + continuation_ids)[dropped:]
    positions = tuple(range(len(causal) - len(continuation_ids) - 1, len(causal) - 1))
    return LikelihoodPlan(
        request_digest=configuration_digest(request.model_dump()),
        identity=request.identity,
        context_token_ids=tuple(context_ids),
        continuation_token_ids=tuple(continuation_ids),
        model_input_token_ids=tuple(causal[:-1]),
        continuation_logit_positions=positions,
        left_context_tokens_dropped=dropped,
        trailing_context_characters_moved=moved,
    )


def validate_likelihood(
    plan: LikelihoodPlan, observation: LikelihoodObservation, *, capability: bool = False
) -> float:
    """Validate a captured backend observation; capability defaults unavailable.

    A true flag is only an admission input after independent backend qualification,
    not evidence itself. This function performs no inference and installs no backend.
    """
    if capability is not True:
        raise LikelihoodUnavailable("Raw conditional-likelihood backend has not been qualified")
    for field in (
        "request_digest",
        "identity",
        "model_input_token_ids",
        "continuation_token_ids",
        "continuation_logit_positions",
    ):
        if getattr(plan, field) != getattr(observation, field):
            raise LikelihoodUnavailable(f"Backend observation disagrees on {field}")
    values = observation.token_logprobs
    if len(values) != len(plan.continuation_token_ids) or not all(math.isfinite(v) and v <= 0 for v in values):
        raise LikelihoodUnavailable("Missing, nonfinite or positive continuation log probabilities")
    total = math.fsum(values)
    if not math.isfinite(observation.summed_logprob) or not math.isclose(
        total, observation.summed_logprob, rel_tol=1e-7, abs_tol=1e-8
    ):
        raise LikelihoodUnavailable("Continuation sum mismatch")
    return total
