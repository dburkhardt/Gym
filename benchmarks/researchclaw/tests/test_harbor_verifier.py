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
    (source / "evaluation").mkdir(parents=True)
    (source / "evaluation/score.py").write_text("# test-only source fixture")
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

    async def spawn(args, **kwargs):
        seen.append(args)
        assert (tmp_path / "verifier/creator-scoring/identity.json").exists()
        from pathlib import Path

        Path(args[-1]).write_text(json.dumps({"failures": {}, "reward": 0.75}))

    monkeypatch.setattr(module, "run_controller", spawn)
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

    monkeypatch.setattr(module, "run_controller", forbidden)
    with pytest.raises(ValueError, match="instructions differ"):
        await verifier.verify()


@pytest.mark.asyncio
async def test_changed_reference_refused_before_controller(tmp_path, monkeypatch):
    verifier, _ = setup(tmp_path, monkeypatch)
    (tmp_path / "source/tasks/Life_002/target_study/checklist.json").write_text("[]")

    async def forbidden(*args, **kwargs):
        raise AssertionError("must not dispatch with changed scientific reference")

    monkeypatch.setattr(module, "run_controller", forbidden)
    with pytest.raises(ValueError, match="reference mismatch"):
        await verifier.verify()


@pytest.mark.asyncio
async def test_submitted_symlink_refused_before_controller(tmp_path, monkeypatch):
    verifier, submitted = setup(tmp_path, monkeypatch)
    secret = tmp_path / "judge-only-reference.txt"
    secret.write_text("not available to submitted report")
    (submitted / "report.txt").symlink_to(secret)

    async def download(remote, destination):
        shutil.copytree(submitted, destination, symlinks=True)

    async def forbidden(*args, **kwargs):
        raise AssertionError("must not dispatch with submitted symlink")

    monkeypatch.setattr(verifier.environment, "download_dir", download)
    monkeypatch.setattr(module, "run_controller", forbidden)
    with pytest.raises(ValueError, match="non-regular"):
        await verifier.verify()


def test_fifo_rejected_before_any_content_read(tmp_path, monkeypatch):
    import os
    from pathlib import Path

    os.mkfifo(tmp_path / "report.md")
    monkeypatch.setattr(
        Path, "read_bytes", lambda self: (_ for _ in ()).throw(AssertionError("read before validation"))
    )
    with pytest.raises(ValueError, match="non-regular"):
        module.regular_tree_manifest(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["timeout", "cancel", "success_with_child"])
async def test_shared_controller_reaps_real_descendants(tmp_path, mode):
    import asyncio
    import sys

    import psutil

    pidfile = tmp_path / "child.pid"
    script = tmp_path / "controller.py"
    script.write_text(
        "import subprocess,sys,time\nfrom pathlib import Path\nprint('partial-controller-diagnostic',flush=True)\n"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
        f"Path({str(pidfile)!r}).write_text(str(p.pid))\n"
        + ("time.sleep(60)\n" if mode != "success_with_child" else "")
    )
    # The success case must close inherited pipes before parent completion can be observed.
    if mode == "success_with_child":
        script.write_text(
            script.read_text().replace(
                "'import time;time.sleep(60)'])",
                "'import time;time.sleep(60)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)",
            )
        )
    task = asyncio.create_task(
        module.run_controller(
            [sys.executable, str(script)], timeout=0.3 if mode == "timeout" else 10, out=tmp_path, source_root=tmp_path
        )
    )
    if mode == "cancel":
        for _ in range(300):
            if pidfile.exists():
                break
            await asyncio.sleep(0.01)
        assert pidfile.exists()
        task.cancel()
    if mode == "timeout":
        with pytest.raises(TimeoutError):
            await task
    elif mode == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await task
    assert "partial-controller-diagnostic" in (tmp_path / "controller.log").read_text()
    pid = int(pidfile.read_text())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    receipt = json.loads((tmp_path / "controller-lifecycle.json").read_text())
    assert receipt["owned_workspace_absent"]
    assert receipt["completed"] is (mode == "success_with_child")


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_genuine_zero_and_failed_judge_not_conflated(tmp_path, monkeypatch, failed):
    verifier, _ = setup(tmp_path, monkeypatch)

    async def result(command, **kwargs):
        from pathlib import Path

        Path(command[-1]).write_text(
            json.dumps({"failures": {"0": "invalid"} if failed else {}, "reward": None if failed else 0.0})
        )

    monkeypatch.setattr(module, "run_controller", result)
    if failed:
        with pytest.raises(RuntimeError, match="cannot become a zero"):
            await verifier.verify()
    else:
        assert (await verifier.verify()).rewards == {"researchclaw_native_checklist": 0.0}
