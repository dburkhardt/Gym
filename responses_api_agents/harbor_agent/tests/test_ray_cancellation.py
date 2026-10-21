# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from unittest.mock import patch

import pytest

from responses_api_agents.harbor_agent.app import _await_owned_ray_job


@pytest.mark.asyncio
async def test_repeated_request_cancellation_waits_for_owned_cleanup():
    result = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(_await_owned_ray_job(result, settle_timeout_s=1))
    with patch("responses_api_agents.harbor_agent.app.ray.cancel") as cancel:
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not result.cancelled()
        result.set_exception(RuntimeError("remote cancelled after finalization"))
        with pytest.raises(asyncio.CancelledError):
            await task
        cancel.assert_called_once_with(result, force=False, recursive=True)


@pytest.mark.asyncio
async def test_unsettled_job_is_not_force_killed_or_reported_clean(caplog, tmp_path):
    result = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(
        _await_owned_ray_job(result, settle_timeout_s=0.01, receipt_path=tmp_path / "cancel.json")
    )
    with patch("responses_api_agents.harbor_agent.app.ray.cancel") as cancel:
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not result.done()
        assert "cleanup remains unconfirmed" in caplog.text
        receipt = json.loads((tmp_path / "cancel.json").read_text())
        assert receipt["phase"] == "settlement_unknown"
        assert not receipt["cleanup_confirmed"]
        cancel.assert_called_once_with(result, force=False, recursive=True)
    result.set_result("late preserved artifact")
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_normal_remote_failure_preserved():
    result = asyncio.get_running_loop().create_future()
    result.set_exception(ValueError("native failure"))
    with pytest.raises(ValueError, match="native failure"):
        await _await_owned_ray_job(result, settle_timeout_s=1)


@pytest.mark.asyncio
async def test_cancel_rpc_failure_preserves_request_cancellation(caplog):
    result = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(_await_owned_ray_job(result, settle_timeout_s=0.01))
    with patch("responses_api_agents.harbor_agent.app.ray.cancel", side_effect=RuntimeError("RPC lost")):
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert "cleanup remains unconfirmed" in caplog.text
    result.set_result("late")
    await asyncio.sleep(0)
