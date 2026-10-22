# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from types import SimpleNamespace

import pytest
from aiohttp import web

from benchmarks.naturebench.unix_bridge import verifier_unix_bridge


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["startup", "cleanup"])
async def test_cancellation_settles_owned_start_and_cleanup(tmp_path, phase):
    directory = tmp_path / ("deep" * 35) / "gym-judge"
    directory.mkdir(parents=True)
    started, release_start, cleaning, release_cleanup = [asyncio.Event() for _ in range(4)]
    completed = []

    async def execute(*, command, timeout_sec):
        if command.startswith("nohup"):
            started.set()
            await release_start.wait()
            (directory / "relay-port.json").write_text(json.dumps({"port": 12345, "pid": 123}))
            completed.append("start")
            return SimpleNamespace(return_code=0, stdout="123\n")
        assert "/proc" in command and "/logs/verifier/gym-judge/relay.py" in command
        cleaning.set()
        await release_cleanup.wait()
        completed.append("cleanup")
        return SimpleNamespace(return_code=0, stdout='{"before":[123],"after":[]}')

    environment = SimpleNamespace(capabilities=SimpleNamespace(mounted=True), exec=execute)
    runner = web.AppRunner(web.Application())
    await runner.setup()

    async def run():
        async with verifier_unix_bridge(runner, environment=environment, directory=directory):
            assert (directory / "model.sock").exists()

    task = asyncio.create_task(run())
    try:
        await started.wait()
        if phase == "startup":
            task.cancel()
        release_start.set()
        await cleaning.wait()
        task.cancel()  # A second cancellation must not abandon cleanup.
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert completed == ["start", "cleanup"]
        assert not (directory / "model.sock").exists()
    finally:
        release_start.set()
        release_cleanup.set()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_cleanup_failure_still_unlinks_socket(tmp_path):
    async def execute(*, command, timeout_sec):
        if command.startswith("nohup"):
            (tmp_path / "relay-port.json").write_text(json.dumps({"port": 12345, "pid": 123}))
            return SimpleNamespace(return_code=0, stdout="123")
        import shlex

        compile(shlex.split(command)[2], "cleanup", "exec")
        return SimpleNamespace(return_code=1, stdout="fixture cleanup failure")

    environment = SimpleNamespace(capabilities=SimpleNamespace(mounted=True), exec=execute)
    runner = web.AppRunner(web.Application())
    await runner.setup()
    try:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            async with verifier_unix_bridge(runner, environment=environment, directory=tmp_path):
                pass
        assert not (tmp_path / "model.sock").exists()
        assert (tmp_path / "relay-cleanup-error.json").exists()
    finally:
        await runner.cleanup()
