# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from nemo_gym.base_responses_api_model import CaptureStore
from responses_api_agents.moosechem_native.artifacts import admit_attempt, attempt_path


def previous(root, task, key, capture):
    out, identity = admit_attempt(root, task, key, capture)
    out.mkdir(parents=True)
    (out / "execution-identity.jsonl").write_text(json.dumps({"task": task, **identity}) + "\n")
    return out


def test_repeats_attempts_unknown_and_duplicate(tmp_path):
    root, capture = tmp_path / "artifacts", tmp_path / "captures"
    store = CaptureStore(capture, provider_dispatch_guard=True)
    store.identity()
    out = previous(root, 18, "task18-0", capture)
    store.write_dispatch("task18-0", "unknown", {"provider_attempts": [{"outcome": "unknown"}]})
    # An independent repeat does not retry an unresolved earlier logical rollout.
    other = previous(root, 18, "task18-1", capture)
    assert out != other
    with pytest.raises(Exception, match="unresolved"):
        admit_attempt(root, 18, "task18-0-a1", capture)
    with pytest.raises(FileExistsError):
        admit_attempt(root, 18, "task18-1", capture)
    for name in ("result.json", "screening.json", "model-traces.jsonl"):
        (other / name).write_text("preserved")
    (other / "native-execution.jsonl").write_text('{"return_code":0,"error_type":null}\n')
    before = {p.name: p.read_bytes() for p in other.iterdir()}
    fresh, identity = admit_attempt(root, 18, "task18-1-a1", capture)
    assert fresh != other and identity["episode"]["attempt"] == 1
    assert before == {p.name: p.read_bytes() for p in other.iterdir()}


def test_legacy_missing_store_and_bad_identity_fail_before_creation(tmp_path):
    root, capture = tmp_path / "artifacts", tmp_path / "captures"
    with pytest.raises(Exception):
        admit_attempt(root, 18, "rollout", capture)
    assert not root.exists() and not capture.exists()
    for task, key in [(18, "../escape"), (-1, "valid"), ("../escape", "valid")]:
        with pytest.raises(ValueError):
            attempt_path(root, task, key)
    (root / "task18").mkdir(parents=True)
    with pytest.raises(ValueError, match="Legacy"):
        admit_attempt(root, 18, "rollout", capture)


def test_same_logical_partial_execution_or_changed_store_refused(tmp_path):
    root, capture = tmp_path / "artifacts", tmp_path / "captures"
    store = CaptureStore(capture, provider_dispatch_guard=True)
    store.identity()
    out = previous(root, 18, "run", capture)
    with pytest.raises(FileNotFoundError):
        admit_attempt(root, 18, "run-a1", capture)
    (out / "native-execution.jsonl").write_text('{"return_code":1,"error_type":"timeout"}\n')
    with pytest.raises(ValueError, match="requires reconciliation"):
        admit_attempt(root, 18, "run-a1", capture)
    (capture / ".capture-store-identity").write_text("f" * 32)
    with pytest.raises(ValueError, match="capture store changed"):
        admit_attempt(root, 18, "run-a1", capture)


@pytest.mark.asyncio
async def test_native_execution_returns_exact_directory_and_persisted_lineage(tmp_path, monkeypatch):
    import shlex
    import sys
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from nemo_gym.server_utils import ServerClient
    from responses_api_agents.moosechem_native import app

    capture = tmp_path / "capture"
    CaptureStore(capture, provider_dispatch_guard=True).identity()
    global_config = {
        "provider_dispatch_guard": True,
        "policy_model_name": app.MODEL,
        "model_call_capture_dir": str(capture),
    }
    monkeypatch.setattr(app, "get_global_config_dict", lambda: global_config)
    monkeypatch.setattr(app, "get_first_server_config_dict", lambda *_: {"host": "127.0.0.1", "port": 1})
    seen = []

    @asynccontextmanager
    async def workspace(provider):
        yield SimpleNamespace(raw={"workspace": tmp_path})

    @asynccontextmanager
    async def bridge(*args, **kwargs):
        seen.append(kwargs)
        yield

    class Provider:
        def __init__(self, **kwargs):
            pass

        async def exec(self, handle, command, **kwargs):
            argv = shlex.split(command)
            out = app.Path(argv[argv.index("--output") + 1])
            assert kwargs["env"]["MOOSE_LINEAGE"] == seen[-1]["lineage"]
            (out / "result.json").write_text('{"scope":"fixture","evaluation":[]}')
            (out / "screening.json").write_text(json.dumps([{}, {"question": [[1, 1]] * 4}]))
            (out / "model-traces.jsonl").write_text(json.dumps({"request": {"model": app.MODEL}}) + "\n")
            return SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")

    monkeypatch.setattr(app, "scoring_workspace", workspace)
    monkeypatch.setattr(app, "bridge", bridge)
    monkeypatch.setattr(app, "LocalProvider", Provider)
    agent = app.Agent(
        config=app.Config(
            host="127.0.0.1",
            port=8080,
            name="moose",
            entrypoint="app.py",
            native_python=sys.executable,
            upstream_repo=tmp_path,
            output_root=tmp_path / "out",
            frozen_tasks=tmp_path / "tasks",
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    body = app.RunRequest.model_validate(
        {"responses_create_params": {"input": []}, "verifier_metadata": {"id": 18}, "_ng_rollout_id": "native-0"}
    )
    out = await agent.execute_native(18, body)
    assert out == attempt_path(agent.config.output_root, 18, "native-0")
    assert agent.read_result(18, out)[3] == 1
    assert seen[0]["lineage"] == "native-0"
    identity = json.loads((out / "execution-identity.jsonl").read_text())
    assert identity["capture_key"] == "native-0"
    assert any(key.endswith("/artifacts.py") for key in identity["source"])
    assert json.loads((out / "native-execution.jsonl").read_text())["return_code"] == 0
    with pytest.raises(FileExistsError):
        await agent.execute_native(18, body)
    assert len(seen) == 1

    CaptureStore(capture, provider_dispatch_guard=True).write_dispatch(
        "native-0", "unresolved", {"provider_attempts": [{"outcome": "unknown"}]}
    )
    same_logical = app.RunRequest.model_validate(body.model_dump() | {"_ng_rollout_id": "native-0-a1"})
    with pytest.raises(Exception, match="unresolved"):
        await agent.execute_native(18, same_logical)
    independent = app.RunRequest.model_validate(body.model_dump() | {"_ng_rollout_id": "native-1"})
    separate = await agent.execute_native(18, independent)
    assert separate != out and seen[-1]["lineage"] == "native-1" and len(seen) == 2
