# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientResponseError
from pydantic import ValidationError

from responses_api_models.openai_model.app import UpstreamAdmissionTimeoutError
from responses_api_models.openai_model.tests.test_cooldown import server


@pytest.mark.asyncio
async def test_cooldown_exhaustion_never_dispatches():
    model = server()
    model.config.upstream_admission_timeout_seconds = 0.02
    model._rate_limit_until = time.monotonic() + 1
    operation = AsyncMock()
    with pytest.raises(UpstreamAdmissionTimeoutError):
        await model._call_upstream(operation)
    operation.assert_not_awaited()
    assert not model._semaphore.locked()


@pytest.mark.asyncio
async def test_wait_then_full_provider_execution_is_not_timed_by_queue():
    model = server()
    model.config.upstream_admission_timeout_seconds = 0.1
    model._rate_limit_until = time.monotonic() + 0.02

    async def operation():
        # Even longer than the entire queue allowance is permitted; the client's
        # independent provider timeout remains responsible for this phase.
        await asyncio.sleep(0.15)
        return "ok"

    assert await model._call_upstream(operation) == "ok"
    assert not model._semaphore.locked()


@pytest.mark.asyncio
async def test_shared_peer_extension_and_slot_wait_share_budget(tmp_path):
    from responses_api_models.openai_model.shared_cooldown import SharedCooldown

    model = server()
    model.config.upstream_admission_timeout_seconds = 0.06
    peer = SharedCooldown(tmp_path / "cooldown", "test", "http://localhost", "test")
    model._shared_cooldown = SharedCooldown(tmp_path / "cooldown", "test", "http://localhost", "test")
    await model._semaphore.acquire()
    operation = AsyncMock()
    task = asyncio.create_task(model._call_upstream(operation))
    await asyncio.sleep(0.025)
    peer.extend(0.05)
    model._semaphore.release()
    with pytest.raises(UpstreamAdmissionTimeoutError):
        await task
    operation.assert_not_awaited()
    assert not model._semaphore.locked()


@pytest.mark.asyncio
async def test_multiple_rejected_attempts_do_not_reset_budget():
    model = server()
    model.config.upstream_admission_timeout_seconds = 0.045
    model.config.upstream_retry_policy.max_attempts = 4
    calls = []

    async def operation():
        calls.append(1)
        raise ClientResponseError(None, (), status=429, headers={"Retry-After": "0.03"})

    with pytest.raises(UpstreamAdmissionTimeoutError):
        await model._call_upstream(operation)
    assert calls == [1, 1]
    assert not model._semaphore.locked()


@pytest.mark.asyncio
async def test_cancel_queued_wait_preserves_other_owner_slot():
    model = server()
    model.config.upstream_admission_timeout_seconds = 1
    await model._semaphore.acquire()
    operation = AsyncMock()
    task = asyncio.create_task(model._call_upstream(operation))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    operation.assert_not_awaited()
    assert model._semaphore.locked()
    model._semaphore.release()
    assert await model._call_upstream(AsyncMock(return_value="ok")) == "ok"


@pytest.mark.asyncio
async def test_non_cooldown_retry_backoff_is_bounded_not_retried_after_expiry():
    model = server()
    model.config.upstream_rate_limit_cooldown_seconds = None
    model.config.upstream_admission_timeout_seconds = 0.01
    model.config.upstream_retry_policy.backoff_initial_seconds = 0.1
    operation = AsyncMock(side_effect=ClientResponseError(SimpleNamespace(real_url="http://fixture"), (), status=429))
    with pytest.raises(UpstreamAdmissionTimeoutError):
        await model._call_upstream(operation)
    assert operation.await_count == 1


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_admission_budget(value):
    config = server().config
    with pytest.raises(ValidationError):
        type(config).model_validate(config.model_dump() | {"upstream_admission_timeout_seconds": value})


@pytest.mark.asyncio
async def test_jitter_consumes_budget_before_any_provider_attempt():
    model = server()
    model.config.upstream_admission_timeout_seconds = 0.01
    model.config.upstream_retry_policy.pre_request_jitter_seconds = (0.1, 0.1)
    operation = AsyncMock()
    with pytest.raises(UpstreamAdmissionTimeoutError):
        await model._call_upstream(operation)
    operation.assert_not_awaited()
    assert not model._semaphore.locked()
