# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientResponseError
from pydantic import ValidationError

from responses_api_models.openai_model.app import UpstreamAdmissionTimeoutError
from responses_api_models.openai_model.shared_cooldown import SharedCooldown
from responses_api_models.openai_model.tests.test_cooldown import server


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value,pressure",
    [
        ("0", True),
        ("-123", True),
        ("12", False),
        (None, False),
        ("bad", False),
        ("NaN", False),
        ("inf", False),
        ("-inf", False),
    ],
)
async def test_header_pressure_is_explicit_and_finite(value, pressure):
    model = server()
    model.config.upstream_exhausted_token_cooldown_seconds = 1
    headers = {"X-RateLimit-Remaining-Tokens": value} if value is not None else {}
    await model._observe_quota_headers(200, headers)
    assert (model._rate_limit_until > time.monotonic()) is pressure


@pytest.mark.asyncio
async def test_disabled_and_error_headers_do_not_change_admission():
    model = server()
    await model._observe_quota_headers(200, {"x-ratelimit-remaining-tokens": "-1"})
    assert model._rate_limit_until == 0
    model.config.upstream_exhausted_token_cooldown_seconds = 1
    await model._observe_quota_headers(429, {"x-ratelimit-remaining-tokens": "-1"})
    assert model._rate_limit_until == 0  # Existing observed429 handler remains authoritative.


@pytest.mark.asyncio
async def test_success_observation_no_resend_and_shared_peer_deadline(tmp_path, monkeypatch):
    model = server()
    model.config.upstream_exhausted_token_cooldown_seconds = 1
    model._client.set_response_headers_observer(observer=model._observe_quota_headers)
    model._shared_cooldown = SharedCooldown(tmp_path, "fixture", "http://unused", "test")
    response = SimpleNamespace(status=200, headers={"x-ratelimit-remaining-tokens": "-10"})
    send = AsyncMock(return_value=response)
    monkeypatch.setattr("nemo_gym.openai_utils.request", send)
    assert await model._call_upstream(lambda: model._client._request_with_retry(url="http://unused")) is response
    send.assert_awaited_once()
    peer = server()
    peer._shared_cooldown = SharedCooldown(tmp_path, "fixture", "http://unused", "test")
    peer.config.upstream_admission_timeout_seconds = 0.01
    operation = AsyncMock()
    with pytest.raises(UpstreamAdmissionTimeoutError):
        await peer._call_upstream(operation)
    operation.assert_not_awaited()
    assert not peer._semaphore.locked()


@pytest.mark.asyncio
async def test_success_then_transport_ambiguity_never_retried(monkeypatch):
    model = server()
    model.config.upstream_exhausted_token_cooldown_seconds = 0.01
    model._client.set_response_headers_observer(observer=model._observe_quota_headers)
    send = AsyncMock(side_effect=TimeoutError("response unknown"))
    monkeypatch.setattr("nemo_gym.openai_utils.request", send)
    with pytest.raises(TimeoutError):
        await model._call_upstream(lambda: model._client._request_with_retry(url="http://unused"))
    send.assert_awaited_once()


@pytest.mark.asyncio
async def test_429_supported_propagation_and_single_attempt():
    model = server()
    model.config.propagate_upstream_http_status_codes = frozenset({429})
    model.config.upstream_retry_policy.max_attempts = 1
    rejected = ClientResponseError(None, (), status=429, headers={"Retry-After": "0.01"})
    operation = AsyncMock(side_effect=rejected)
    with pytest.raises(ClientResponseError) as caught:
        await model._call_upstream(operation)
    assert caught.value is rejected and model._should_propagate_upstream_http_error(rejected)
    operation.assert_awaited_once()
    assert model._rate_limit_until > time.monotonic()


@pytest.mark.parametrize("delay", [0, -1, float("inf"), float("nan")])
def test_invalid_configuration(delay):
    model = server()
    with pytest.raises(ValidationError):
        type(model.config).model_validate(
            model.config.model_dump() | {"upstream_exhausted_token_cooldown_seconds": delay}
        )


def test_requires_existing_admission_policy():
    model = server()
    with pytest.raises(ValidationError):
        type(model.config).model_validate(
            model.config.model_dump()
            | {"upstream_rate_limit_cooldown_seconds": None, "upstream_exhausted_token_cooldown_seconds": 1}
        )


def test_enabled_policy_is_bound_and_changes_admission_evidence():
    from omegaconf import OmegaConf

    from nemo_gym.cli.admission_evidence import resolved_admission_evidence
    from responses_api_models.openai_model.app import SimpleModelServer

    original = server()
    settings = original.config.model_dump(mode="json")
    before = resolved_admission_evidence(
        OmegaConf.create({"policy": {"responses_api_models": {"openai_model": settings}}})
    )
    settings.update(upstream_exhausted_token_cooldown_seconds=60, propagate_upstream_http_status_codes=[429])
    configured = SimpleModelServer(
        config=type(original.config).model_validate(settings), server_client=original.server_client
    )
    assert configured._client._response_headers_observer is not None
    after = resolved_admission_evidence(
        OmegaConf.create({"policy": {"responses_api_models": {"openai_model": settings}}})
    )
    assert before["sha256"] != after["sha256"]
    assert "exhausted_token_cooldown_seconds" not in before["roles"][0]
    assert after["roles"][0]["exhausted_token_cooldown_seconds"] == 60
    assert after["roles"][0]["propagate_upstream_http_status_codes"] == [429]


def test_runtime_observer_preserves_client_schema():
    from nemo_gym.openai_utils import NeMoGymAsyncOpenAI

    assert "response_headers_observer" not in NeMoGymAsyncOpenAI.model_json_schema()["properties"]
