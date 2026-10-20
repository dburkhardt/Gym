# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from types import SimpleNamespace

import pytest

from resources_servers.scicode import managed_execution as execution


@pytest.mark.asyncio
async def test_cancel_during_creation_settles_and_removes(tmp_path, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()
    order = []

    class Sandbox:
        def __init__(self, provider, spec):
            assert provider["docker"]["create"]["network"] == "none"
            assert spec.metadata["run"]

        async def start(self):
            started.set()
            await release.wait()
            order.append("created")

        async def serialize(self):
            return {"sandbox_id": "owned"}

        async def stop(self):
            assert order == ["created"]
            order.append("removed")

    monkeypatch.setattr(execution, "AsyncSandbox", Sandbox)
    task = asyncio.create_task(
        execution.execute_substep(
            program="assert True",
            data_path=tmp_path / "data",
            image="sha256:" + "a" * 64,
            artifact_dir=tmp_path,
            timeout_secs=1,
            cpu=1,
            memory_mib=512,
        )
    )
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert order == ["created", "removed"]
    assert json.loads(next(tmp_path.rglob("lifecycle.json")).read_text())["cleanup_confirmed"]


@pytest.mark.asyncio
async def test_runtime_failure_raises_instead_of_scoring_wrong(tmp_path, monkeypatch):
    calls = []

    class Sandbox:
        def __init__(self, *args):
            pass

        async def start(self):
            pass

        async def serialize(self):
            return {"sandbox_id": "owned"}

        async def exec(self, command, **kwargs):
            calls.append(command)
            return SimpleNamespace(return_code=1, stdout="", stderr="H5 not readable")

        async def stop(self):
            calls.append("removed")

    monkeypatch.setattr(execution, "AsyncSandbox", Sandbox)
    with pytest.raises(RuntimeError, match="runtime/data preflight"):
        await execution.execute_substep(
            program="raise AssertionError",
            data_path=tmp_path / "data",
            image="sha256:" + "a" * 64,
            artifact_dir=tmp_path,
            timeout_secs=1,
            cpu=1,
            memory_mib=512,
        )
    assert len(calls) == 2 and calls[-1] == "removed"
    assert not list(tmp_path.rglob("result.json"))
