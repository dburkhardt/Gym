# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.scivisagentbench.managed import checked_tree, viewer_spec
from benchmarks.scivisagentbench.worker import runtime_identity, stage_judge_data
from responses_api_agents.scivis_native import app


def tree(root):
    root.mkdir()
    (root / "file").write_text("fixture")
    return {"source": str(root), "sha256": {"file": hashlib.sha256(b"fixture").hexdigest()}}


def test_viewer_full_resource_scope_and_no_reference_mount(tmp_path):
    source = tree(tmp_path / "source")
    data = tree(tmp_path / "data")
    packet = {
        "image": "sha256:" + "a" * 64,
        "mounts": [{**source, "target": "/source"}, {**data, "target": "/workspace/data"}],
    }
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = viewer_spec(packet, workspace, "owned", 7200)
    assert spec.resources.memory_mib == 16384 and spec.resources.cpu == 2 and spec.ttl_s == 7380
    assert spec.metadata["run"] == "owned" and not spec.ports
    assert all("references" not in v and "judge" not in v for v in spec.provider_options["volumes"])
    (tmp_path / "source" / "GS").mkdir()
    (tmp_path / "source" / "GS" / "answer").write_text("no")
    with pytest.raises(ValueError, match="Reference"):
        viewer_spec(packet, workspace, "owned", 7200)


def test_mount_tree_rejects_symlink_and_unpinned_file(tmp_path):
    row = tree(tmp_path / "source")
    (tmp_path / "source" / "extra").write_text("unexpected")
    with pytest.raises(ValueError, match="mismatch"):
        checked_tree(Path(row["source"]), row["sha256"])
    (tmp_path / "source" / "extra").unlink()
    (tmp_path / "source" / "link").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symlinks"):
        checked_tree(Path(row["source"]), row["sha256"])


def test_judge_copy_never_follows_solver_links(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "escape").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symlinks"):
        stage_judge_data(workspace, tmp_path / "judge", tmp_path / "refs", {})
    (workspace / "escape").unlink()
    (workspace / "answer").write_text("solver")
    refs = tmp_path / "refs"
    (refs / "GS").mkdir(parents=True)
    (refs / "GS" / "gold").write_text("gold")
    stage_judge_data(workspace, tmp_path / "judge", refs, {"GS/gold": hashlib.sha256(b"gold").hexdigest()})
    assert (tmp_path / "judge" / "GS" / "gold").read_text() == "gold"
    assert not (workspace / "GS").exists()


@pytest.mark.asyncio
async def test_managed_failure_is_masked_not_scientific_zero(tmp_path, monkeypatch):
    import asyncio

    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps({"case1": {}}))
    fake = SimpleNamespace(
        _semaphore=asyncio.Semaphore(1),
        config=SimpleNamespace(
            inference_enabled=True,
            task_manifest=str(manifest),
            output_root=str(tmp_path / "out"),
            model_server=SimpleNamespace(name="policy_model"),
        ),
    )
    monkeypatch.setattr(app, "FROZEN_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest())
    monkeypatch.setattr(
        app, "get_global_config_dict", lambda: {"provider_dispatch_guard": True, "policy_model_name": app.MODEL}
    )
    monkeypatch.setattr(app, "get_first_server_config_dict", lambda *a: {"host": "127.0.0.1", "port": 1})
    monkeypatch.setattr(app, "current_rollout_id", lambda: "fixture")

    async def fail(*args):
        raise TimeoutError("native process deadline")

    monkeypatch.setattr(app, "execute", fail)
    body = app.RunRequest.model_validate(
        {
            "responses_create_params": {"input": []},
            "verifier_metadata": {"task_id": "case1", "frozen_manifest_sha256": app.FROZEN_SHA256},
        }
    )
    result = await app.Agent.run(fake, body)
    assert result.mask_sample and result.scientific_score is None and "TimeoutError" in result.failure_reason
    with pytest.raises(ValueError, match="no implicit retry"):
        await app.Agent.run(fake, body)


def test_runtime_identity_records_actual_imports_and_refuses_stale_source(tmp_path):
    import importlib

    paths = [
        Path(importlib.import_module("benchmarks.scivisagentbench." + name).__file__).resolve()
        for name in ("native", "transport", "judge", "evaluate", "worker")
    ]
    import inspect

    from nemo_gym.sandbox.guarded_chat_transport import GuardedChatTransport

    paths.append(Path(inspect.getfile(GuardedChatTransport)).resolve())
    expected = {
        "source": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "interpreter": {
            "path": str(Path(sys.executable).absolute()),
            "resolved_path": str(Path(sys.executable).resolve()),
            "sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        },
    }
    runtime_identity(tmp_path, expected)
    receipt = json.loads((tmp_path / "runtime-imports.json").read_text())
    assert receipt["source"] == expected["source"]
    # Publication is an immutable attempt record, so use a fresh output for refusal.
    changed = tmp_path / "changed"
    changed.mkdir()
    expected["source"][str(paths[0])] = "wrong"
    with pytest.raises(ValueError, match="identity mismatch"):
        runtime_identity(changed, expected)
    assert not (changed / "provider-calls").exists()


def test_viewer_rejects_symlink_mount_root(tmp_path):
    row = tree(tmp_path / "real")
    (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)
    packet = {
        "image": "sha256:" + "a" * 64,
        "mounts": [{**row, "source": str(tmp_path / "alias"), "target": "/source"}],
    }
    with pytest.raises(ValueError, match="Qualified tree"):
        viewer_spec(packet, tmp_path / "workspace", "run", 10)


def test_judge_rejects_symlink_reference_ancestor(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "gold").write_text("gold")
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "GS").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="reference mismatch"):
        stage_judge_data(workspace, tmp_path / "judge", refs, {"GS/gold": hashlib.sha256(b"gold").hexdigest()})


@pytest.mark.asyncio
async def test_request_manifest_binding_rejected_before_execution(tmp_path):
    import asyncio

    fake = SimpleNamespace(_semaphore=asyncio.Semaphore(1), config=SimpleNamespace(inference_enabled=True))
    body = app.RunRequest.model_validate(
        {
            "responses_create_params": {"input": []},
            "verifier_metadata": {"task_id": "napari-case1", "frozen_manifest_sha256": "wrong"},
        }
    )
    with pytest.raises(ValueError, match="not bound"):
        await app.Agent.run(fake, body)


@pytest.mark.asyncio
async def test_judge_deadline_settles_accepted_call_and_blocks_later_dispatch(tmp_path, monkeypatch):
    import asyncio

    import httpx

    from benchmarks.scivisagentbench import worker

    observed = []

    class Transport(httpx.AsyncBaseTransport):
        def __init__(self, **kwargs):
            self.failed = False

        async def handle_async_request(self, request):
            assert request.extensions["timeout"]["read"] == 0.4
            observed.append("send")
            await asyncio.sleep(0.12)
            return httpx.Response(200, json={"model": worker.MODEL, "choices": []}, request=request)

        async def aclose(self):
            observed.append("closed")

    monkeypatch.setattr(worker, "GuardedChatTransport", Transport)
    monkeypatch.setattr(worker, "runtime_identity", lambda *args: None)
    monkeypatch.setattr(worker, "stage_judge_data", lambda *args: None)

    async def solver(*args):
        return {"success": True}

    monkeypatch.setattr(worker, "run_native", solver)

    async def evaluate(source, case, case_dir, data_dir, mode, sender, output):
        sender({"model": worker.MODEL})
        with pytest.raises(RuntimeError, match="phase stopped"):
            sender({"model": worker.MODEL})
        observed.append("judge-settled")
        return {}

    monkeypatch.setattr(worker, "evaluate", evaluate)
    (tmp_path / "identity.json").write_text(json.dumps({"timeout_s": 0.03, "max_tokens": 16384, "request_timeout_s": 0.4}))
    (tmp_path / "worker-input.json").write_text(
        json.dumps(
            {
                "source_root": "unused",
                "native_config": {},
                "reference_root": "unused",
                "reference_sha256": {},
                "task": {
                    "source_case": {},
                    "judge_case_dir": "unused",
                    "judge_data_dir": "unused",
                    "agent_mode": "native",
                },
            }
        )
    )
    for key in ("SCIVIS_SOCKET", "SCIVIS_TOKEN", "SCIVIS_LINEAGE", "SCIVIS_DIGEST"):
        monkeypatch.setenv(key, "fixture")
    with pytest.raises(TimeoutError):
        await worker.main(tmp_path)
    assert observed == ["send", "judge-settled", "closed"]
    assert not (tmp_path / "worker-result.json").exists()


def test_native_worker_import_does_not_load_sandbox_stack():
    script = """
import importlib.abc
import sys
class RejectSandbox(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'nemo_gym.sandbox' or fullname.startswith('nemo_gym.sandbox.'):
            raise ImportError('native worker must not depend on sandbox runtime')
sys.meta_path.insert(0, RejectSandbox())
from benchmarks.scivisagentbench.worker import GuardedChatTransport
from nemo_gym.guarded_chat_transport import GuardedChatTransport as shared
assert GuardedChatTransport is shared
"""
    subprocess.run([sys.executable, "-c", script], check=True, cwd=Path(__file__).resolve().parents[2])


def test_viewer_workspace_survives_restrictive_umask_without_changing_inputs(tmp_path):
    import os

    from benchmarks.scivisagentbench.managed import create_viewer_workspace

    refs = tmp_path / "references"
    refs.mkdir(mode=0o700)
    previous = os.umask(0o077)
    try:
        workspace = create_viewer_workspace(tmp_path / "attempt")
    finally:
        os.umask(previous)
    assert workspace.stat().st_mode & 0o777 == 0o777
    assert refs.stat().st_mode & 0o777 == 0o700
    with pytest.raises(FileExistsError):
        create_viewer_workspace(tmp_path / "attempt")


def test_native_python_preserves_real_virtual_environment(tmp_path):
    import subprocess
    import venv

    from benchmarks.scivisagentbench.managed import validate_native_python

    environment = tmp_path / "controller"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    executable = environment / "bin" / "python"
    assert executable.is_symlink()
    selected = validate_native_python(executable)
    assert selected == executable.absolute()
    assert selected != executable.resolve()
    site = Path(subprocess.check_output(
        [str(selected), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True,
    ).strip())
    (site / "scivis_venv_only.py").write_text("VALUE = 'controller dependency'\n")
    actual = json.loads(subprocess.check_output(
        [str(selected), "-c", "import json,sys,scivis_venv_only as dep; print(json.dumps([sys.prefix,dep.VALUE]))"],
        text=True,
    ))
    assert actual == [str(environment), "controller dependency"]
    bypassed = subprocess.run(
        [str(executable.resolve()), "-I", "-c", "import scivis_venv_only"], capture_output=True, text=True,
    )
    assert bypassed.returncode != 0
    assert "ModuleNotFoundError" in bypassed.stderr
