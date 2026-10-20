# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import shutil
from types import SimpleNamespace

import pytest


pytest.importorskip("harbor")
from responses_api_agents.harbor_agent.custom_verifiers.researchclaw import verifier as module


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {"provider_dispatch_guard": True})
    submitted = tmp_path / "original"
    submitted.mkdir()
    (submitted / "INSTRUCTIONS.md").write_text("frozen prompt")
    (submitted / "_meta.json").write_text(json.dumps({"task_id": "Life_002"}))
    source = tmp_path / "source"
    refs = source / "tasks/Life_002/target_study"
    refs.mkdir(parents=True)
    (refs / "checklist.json").write_text('[{"content":"criterion"}]')
    manifest = {
        "tasks": {
            "Life_002": {
                "prompt_sha256": hashlib.sha256((submitted / "INSTRUCTIONS.md").read_bytes()).hexdigest(),
                "references": {
                    "target_study/checklist.json": hashlib.sha256((refs / "checklist.json").read_bytes()).hexdigest()
                },
            }
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    class Environment:
        async def download_dir(self, remote, dest):
            assert remote == "/workspace"
            shutil.copytree(submitted, dest)

    verifier = module.ResearchClawVerifier(
        source_root=str(source),
        frozen_manifest=str(manifest_path),
        frozen_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        gym_model_api_base="http://localhost/ng-rollout/task/v1",
        gym_model_name="nvidia/nvidia/nemotron-3.5-super-vl",
        controller_python="python",
        task=SimpleNamespace(
            paths=SimpleNamespace(task_dir=tmp_path / "Life_002"),
            config=SimpleNamespace(verifier=SimpleNamespace(timeout_sec=10)),
        ),
        trial_paths=SimpleNamespace(verifier_dir=tmp_path / "verifier"),
        environment=Environment(),
    )
    return verifier, submitted


@pytest.mark.asyncio
async def test_only_native_scientific_reward_is_returned(tmp_path, monkeypatch):
    verifier, _ = setup(tmp_path, monkeypatch)
    seen = []

    async def spawn(*args, **kwargs):
        seen.append(args)
        assert (tmp_path / "verifier/creator-scoring/identity.json").exists()
        from pathlib import Path

        Path(args[-1]).write_text(json.dumps({"failures": {}, "reward": 0.75}))

        class Process:
            returncode = 0

            async def wait(self):
                return 0

        return Process()

    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", spawn)
    result = await verifier.verify()
    assert result.rewards == {"researchclaw_native_checklist": 0.75}
    assert len(seen) == 1
    with pytest.raises(FileExistsError):
        await verifier.verify()
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_changed_instruction_refused_before_controller(tmp_path, monkeypatch):
    verifier, submitted = setup(tmp_path, monkeypatch)
    (submitted / "INSTRUCTIONS.md").write_text("altered prompt")

    async def forbidden(*args, **kwargs):
        raise AssertionError("must not dispatch")

    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", forbidden)
    with pytest.raises(ValueError, match="instructions differ"):
        await verifier.verify()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_during_spawn", [False, True])
async def test_real_controller_is_reaped_on_cancellation(tmp_path, monkeypatch, cancel_during_spawn):
    import asyncio
    import sys

    verifier, _ = setup(tmp_path, monkeypatch)
    actual_spawn = asyncio.create_subprocess_exec
    spawned = asyncio.Event()
    release = asyncio.Event()
    processes = []

    async def spawn(*args, **kwargs):
        process = await actual_spawn(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
        processes.append(process)
        spawned.set()
        if cancel_during_spawn:
            await release.wait()
        return process

    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", spawn)
    task = asyncio.create_task(verifier.verify())
    try:
        await asyncio.wait_for(spawned.wait(), timeout=5)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)
        assert len(processes) == 1
        assert processes[0].returncode is not None
        assert not (tmp_path / "verifier/creator-scoring/scientific-score.json").exists()
    finally:
        release.set()
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.wait()
