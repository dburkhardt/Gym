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


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_error", [None, "ArtifactError"])
async def test_cancelled_objectref_waits_for_artifact_finalizer_receipt(tmp_path, cleanup_error):
    from responses_api_agents.harbor_agent.app import _write_cancellation_receipt

    result = asyncio.get_running_loop().create_future()
    collecting, release = asyncio.Event(), asyncio.Event()
    receipt = tmp_path / "gym-cancellation-request.json"

    async def worker_finalizer():
        collecting.set()
        await release.wait()
        (tmp_path / "preserved-artifact.txt").write_text("original scientific output")
        _write_cancellation_receipt(
            tmp_path / "gym-cancellation-worker.json",
            {
                "phase": "native_coroutine_settled",
                "cleanup_error": cleanup_error,
                "docker_cleanup_confirmed": False,
            },
        )

    worker = asyncio.create_task(worker_finalizer())
    task = asyncio.create_task(_await_owned_ray_job(result, settle_timeout_s=1, receipt_path=receipt))
    with patch("responses_api_agents.harbor_agent.app.ray.cancel", side_effect=lambda *a, **k: result.cancel()):
        await collecting.wait()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0.02)
        task.cancel()
        await asyncio.sleep(0.02)
        assert result.cancelled() and not task.done()
        assert not (tmp_path / "preserved-artifact.txt").exists()
        release.set()
        await worker
        with pytest.raises(asyncio.CancelledError):
            await task
    data = json.loads(receipt.read_text())
    assert data["phase"] == ("native_finalizer_error" if cleanup_error else "native_finalizer_settled")
    assert not data["cleanup_confirmed"] and not data["settlement_timed_out"]
    assert (tmp_path / "preserved-artifact.txt").read_text() == "original scientific output"


@pytest.mark.asyncio
async def test_cancelled_objectref_missing_worker_receipt_times_out_unconfirmed(tmp_path):
    result = asyncio.get_running_loop().create_future()
    receipt = tmp_path / "gym-cancellation-request.json"
    task = asyncio.create_task(_await_owned_ray_job(result, settle_timeout_s=0.03, receipt_path=receipt))
    with patch("responses_api_agents.harbor_agent.app.ray.cancel", side_effect=lambda *a, **k: result.cancel()):
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    data = json.loads(receipt.read_text())
    assert data["phase"] == "settlement_unknown" and data["settlement_timed_out"]
    assert not data["cleanup_confirmed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt_value", [[], {"phase": "native_coroutine_settled", "cleanup_error": None,
    "timestamp": "2000-01-01T00:00:00+00:00"}])
async def test_stale_or_malformed_worker_receipt_cannot_settle(tmp_path, receipt_value):
    if isinstance(receipt_value, dict):
        receipt_value["job_directory"] = str(tmp_path)
    (tmp_path / "gym-cancellation-worker.json").write_text(json.dumps(receipt_value))
    result = asyncio.get_running_loop().create_future()
    receipt = tmp_path / "gym-cancellation-request.json"
    task = asyncio.create_task(_await_owned_ray_job(result, settle_timeout_s=0.02, receipt_path=receipt))
    with patch("responses_api_agents.harbor_agent.app.ray.cancel", side_effect=lambda *a, **k: result.cancel()):
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert json.loads(receipt.read_text())["phase"] == "settlement_unknown"
