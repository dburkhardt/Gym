# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import hashlib
import json
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from harbor.verifier.verifier import Verifier

from benchmarks.naturebench import prepare as preparation
from benchmarks.naturebench.verifier import MODEL, NatureBenchVerifier


def test_prepare_preserves_original(monkeypatch, tmp_path):
    source = tmp_path / "frozen.jsonl"
    row = {
        "instance_id": "naturebench::paper-1",
        "responses_create_params": {"input": [], "temperature": 0},
        "extra": 42,
    }
    source.write_text(json.dumps(row) + "\n")
    monkeypatch.setenv("NATUREBENCH_FROZEN_SAMPLE", str(source))
    monkeypatch.setattr(preparation, "__file__", str(tmp_path / "prepare.py"))
    result = preparation.prepare()
    actual = json.loads(result.read_text())
    assert all(actual[k] == v for k, v in row.items())
    assert actual["task_id"] == row["instance_id"]
    receipt = json.loads(result.with_suffix(".source.json").read_text())
    assert receipt["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        preparation.prepare()


@pytest.mark.asyncio
async def test_native_verifier_bridge_binding_and_cleanup(monkeypatch, tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "judge_readme.py").write_text("Campaign judge model/provider is not allowed")
    captured = []

    async def capture(request):
        captured.append((dict(request.headers), await request.json()))
        return web.json_response({"id": "mock", "model": MODEL, "status": "completed", "output": []})

    app = web.Application()
    app.router.add_post("/ng-rollout/test/v1/responses", capture)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    verifier = object.__new__(NatureBenchVerifier)
    verifier.task = SimpleNamespace(
        paths=SimpleNamespace(tests_dir=tests), config=SimpleNamespace(model_dump=lambda **_: {})
    )
    verifier.trial_paths = SimpleNamespace(verifier_dir=tmp_path / "out")
    verifier.endpoint = f"http://127.0.0.1:{port}/ng-rollout/test/v1/responses"
    verifier.bridge_host = "127.0.0.1"
    verifier.lock = asyncio.Lock()
    verifier.override_env = {}

    async def native(self):
        async with aiohttp.ClientSession() as client:
            url = self.override_env["JUDGE_BASE_URL"] + "/responses"
            async with client.post(url, json={"model": MODEL}) as rejected:
                assert rejected.status == 401
            headers = {"Authorization": "Bearer " + self.override_env["JUDGE_API_KEY"]}
            async with client.post(url, json={"model": MODEL, "input": "rubric"}, headers=headers) as accepted:
                assert accepted.status == 200
            async with client.post(url, json={"model": MODEL, "input": "changed"}, headers=headers) as rejected:
                assert rejected.status == 409
        return "native-result"

    monkeypatch.setattr(Verifier, "verify", native)
    try:
        assert await verifier.verify() == "native-result"
        assert len(captured) == 1
        assert captured[0][0]["x-nemo-gym-logical-call-id"].endswith(":judge:0")
        assert len(captured[0][0]["x-nemo-gym-call-config-sha256"]) == 64
        assert (tmp_path / "out/gym-judge/response.json").exists()
        with pytest.raises(FileExistsError):
            await verifier.verify()
        async with aiohttp.ClientSession() as client:
            with pytest.raises(aiohttp.ClientConnectorError):
                await client.post(verifier.override_env["JUDGE_BASE_URL"] + "/responses")
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_timer_failure_is_not_budget_exhaustion():
    from benchmarks.naturebench.timed_agent import NatureBenchTimedAgent

    agent = object.__new__(NatureBenchTimedAgent)
    agent.timeout = 14400
    settled = []
    commands = []

    async def inner(*args):
        try:
            await asyncio.Event().wait()
        finally:
            settled.append(True)

    agent.inner = SimpleNamespace(run=inner)

    async def timer(environment, command, timeout_sec=None):
        commands.append(command)
        if command == "wait-expired":
            raise RuntimeError("timer unavailable")

    agent._timer_command = timer
    with pytest.raises(RuntimeError, match="timer unavailable"):
        await agent.run("You have **240 minutes** in total.", None, None)
    assert settled == [True]
    assert commands[-1] == "stop"


def test_materialize_pinned_archive_preserves_science(monkeypatch, tmp_path):
    import tarfile

    from benchmarks.naturebench import prepare_runtime

    source = tmp_path / "original" / "paper"
    (source / "tests").mkdir(parents=True)
    (source / "task.toml").write_text("[agent]\ntimeout_sec=43200\n")
    (source / "tests/judge_readme.py").write_text("original judge")
    (source / "tests/scorer.py").write_text("unchanged scientific scorer")
    archive = tmp_path / "paper.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="paper")
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(prepare_runtime, "patch_judge", lambda text: text + " explicit model bridge")
    target = tmp_path / "prepared" / "paper"
    prepare_runtime.materialize_archive(archive, target, expected_sha256=expected)
    assert (target / "tests/scorer.py").read_bytes() == (source / "tests/scorer.py").read_bytes()
    assert (target / "task.toml").read_bytes() == (source / "task.toml").read_bytes()
    receipt = json.loads((target.parent / "paper.source.json").read_text())
    assert receipt["source_files"]["tests/scorer.py"] == receipt["prepared_files"]["tests/scorer.py"]
    with pytest.raises(FileExistsError):
        prepare_runtime.materialize_archive(archive, target, expected_sha256=expected)
    with pytest.raises(ValueError, match="checksum"):
        prepare_runtime.materialize_archive(archive, tmp_path / "bad", expected_sha256="bad")
