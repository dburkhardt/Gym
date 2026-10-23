# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from benchmarks.moosechem.guarded import GuardedTransport
from benchmarks.moosechem.native import MODEL, TransportAbort
from responses_api_agents.moosechem_native import app


def completion():
    return {
        "id": "fixture",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "native answer"}}
        ],
    }


def test_guarded_native_call_preserves_payload_receipts_and_identity(tmp_path, monkeypatch):
    seen = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["follow_redirects"] is False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, *, json: dict, headers):
            assert (tmp_path / "dispatch-intents.jsonl").exists()
            seen.append((url, json, headers))
            return httpx.Response(200, json=completion(), request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "Client", Client)
    transport = GuardedTransport("/unused", "private-token", "run", "digest")
    payload = {
        "model": MODEL,
        "max_tokens": 8192,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "creator prompt"}],
    }
    answer = transport.create("screening", tmp_path / "model-traces.jsonl", 8192, payload)
    assert seen[0][1] == payload and answer.choices[0].message.content == "native answer"
    assert seen[0][2]["x-nemo-gym-logical-call-id"] == "moosechem:run:0"
    trace = (tmp_path / "model-traces.jsonl").read_text()
    assert "private-token" not in trace and json.loads(trace)["response"] == completion()


def test_guarded_failure_poison_prevents_creator_retry(tmp_path, monkeypatch):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            calls.append(1)
            raise httpx.ReadTimeout("unknown outcome")

    monkeypatch.setattr(httpx, "Client", Client)
    transport = GuardedTransport("/unused", "token", "run", "digest")
    for _ in range(2):
        with pytest.raises(TransportAbort):
            transport.create("screening", tmp_path / "trace.jsonl", 8192, {"model": MODEL, "max_tokens": 8192})
    assert len(calls) == 1
    assert json.loads((tmp_path / "transport-failures.jsonl").read_text())["response_captured"] is False


@pytest.mark.asyncio
async def test_unresolved_task_rejected_before_work(tmp_path):
    agent = SimpleNamespace(config=SimpleNamespace(blocked_task_ids=[26]))
    with pytest.raises(ValueError, match="unresolved"):
        await app.Agent.execute_native(agent, 26, None)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_native_cancellation_settles_child_before_bridge_cleanup(tmp_path, monkeypatch):
    from nemo_gym.base_responses_api_model import CaptureStore
    from nemo_gym.sandbox.providers.local.provider import LocalProvider

    # Exercise the retained historical process cleanup helper, never a new-live configuration.
    monkeypatch.setattr(app, "require_isolated_live", lambda *args: None)
    CaptureStore(tmp_path / "captures", provider_dispatch_guard=True).identity()
    pidfile = tmp_path / "child.pid"
    original_exec = LocalProvider.exec

    async def bounded_exec(self, handle, command, **kwargs):
        code = f"import os,time,pathlib;pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(30)"
        import shlex

        return await original_exec(self, handle, shlex.join([sys.executable, "-c", code]), timeout_s=30)

    monkeypatch.setattr(LocalProvider, "exec", bounded_exec)
    cleaned = []

    @asynccontextmanager
    async def bridge(*args, **kwargs):
        try:
            yield
        finally:
            with pytest.raises(ProcessLookupError):
                os.kill(int(pidfile.read_text()), 0)
            cleaned.append(True)

    monkeypatch.setattr(app, "bridge", bridge)
    monkeypatch.setattr(
        app,
        "get_global_config_dict",
        lambda: {
            "provider_dispatch_guard": True,
            "policy_model_name": MODEL,
            "model_call_capture_dir": str(tmp_path / "captures"),
        },
    )
    monkeypatch.setattr(app, "get_first_server_config_dict", lambda *args: {"host": "127.0.0.1", "port": 1})
    monkeypatch.setattr(app, "current_rollout_id", lambda: "fixture")
    config = SimpleNamespace(
        settled_recovery=None,
        isolated_runtime_image=None,
        blocked_task_ids=[],
        output_root=tmp_path / "outputs",
        native_python=Path(sys.executable),
        upstream_repo=tmp_path,
        output_token_budget=8192,
        native_timeout_seconds=30,
        model_server=SimpleNamespace(name="policy_model"),
    )
    task = asyncio.create_task(app.Agent.execute_native(SimpleNamespace(config=config), 18, None))
    async with asyncio.timeout(5):
        while not pidfile.exists():
            await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned == [True]


def test_prepare_preserves_frozen_ids_and_separates_reference(tmp_path, monkeypatch):
    import hashlib

    from benchmarks.moosechem import prepare

    raw = [
        {"id": 18, "question": "Q", "survey": "S", "groundtruth": "private reference"},
        {"id": 26, "question": "Q2", "survey": "S2", "groundtruth": "other reference"},
    ]
    source = tmp_path / "frozen.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in raw))
    monkeypatch.setenv("MOOSE_FROZEN_TASKS", str(source))
    monkeypatch.setenv("MOOSE_FROZEN_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(prepare, "__file__", str(tmp_path / "prepare.py"))
    output = prepare.prepare()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["task_id"] for r in rows] == ["18", "26"]  # preserve unresolved identity, never silently resample
    assert rows[0]["responses_create_params"]["input"][0]["content"] == "Q\n\nS"
    assert rows[0]["verifier_metadata"] == raw[0]
    before = output.read_bytes()
    monkeypatch.setenv("MOOSE_FROZEN_SHA256", "wrong")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare.prepare()
    assert output.read_bytes() == before


@pytest.mark.asyncio
async def test_actual_unix_bridge_forwards_native_guard_identity(tmp_path, monkeypatch):
    from nemo_gym.sandbox import model_bridge

    calls = []

    class Response:
        status = 200

        async def read(self):
            return json.dumps(completion()).encode()

        def release(self):
            calls.append("released")

    async def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return Response()

    monkeypatch.setattr(model_bridge, "request", request)
    short_root = tempfile.TemporaryDirectory(dir="/tmp", prefix="moose-")
    socket = Path(short_root.name) / "model.sock"
    async with model_bridge.bridge(
        socket,
        endpoint="http://fixture/ng-rollout/run/v1",
        token="private",
        lineage="run",
        digest="digest",
        namespace="moosechem",
        expected_model=MODEL,
    ):
        transport = GuardedTransport(str(socket), "private", "run", "digest")
        result = await asyncio.to_thread(
            transport.create,
            "screening",
            tmp_path / "trace.jsonl",
            8192,
            {"model": MODEL, "max_tokens": 8192, "messages": []},
        )
        assert result.choices[0].message.content == "native answer"
    assert not socket.exists()
    short_root.cleanup()
    assert calls[0][2]["headers"]["x-nemo-gym-logical-call-id"] == "moosechem:run:0"
    assert calls[0][2]["_max_num_tries"] == 1 and calls[-1] == "released"


def test_runtime_import_receipt_rejects_drift_before_dispatch(tmp_path, monkeypatch):
    import hashlib

    from benchmarks.moosechem import guarded, native

    expected = {
        "source": {
            str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (native.__file__, guarded.__file__)
        },
        "interpreter": {
            "path": str(Path(sys.executable).resolve()),
            "sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        },
    }
    identity = tmp_path / "identity.jsonl"
    identity.write_text(json.dumps(expected) + "\n")
    monkeypatch.setenv("MOOSE_EXECUTION_IDENTITY", str(identity))
    guarded.record_runtime_identity(tmp_path)
    actual = json.loads((tmp_path / "runtime-imports.jsonl").read_text())
    assert actual["source"] == expected["source"]
    expected["source"][str(Path(native.__file__).resolve())] = "wrong"
    identity.write_text(json.dumps(expected) + "\n")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        guarded.record_runtime_identity(tmp_path)
    assert not (tmp_path / "dispatch-intents.jsonl").exists()


@pytest.mark.parametrize("image,budget", [(None, 8192), ("sha256:" + "a" * 64, 16384)])
def test_live_isolation_and_cross_phase_budget_fail_before_dispatch(image, budget, monkeypatch):
    config = SimpleNamespace(
        isolated_runtime_image=image,
        public_input_root=Path("public"),
        public_manifest_sha256="a" * 64,
        verifier_server=SimpleNamespace(name="verifier"),
        output_token_budget=budget,
        native_timeout_seconds=14400,
        output_root=Path("out"),
        upstream_repo=Path("creator"),
    )
    monkeypatch.setattr(
        app,
        "get_first_server_config_dict",
        lambda *args: {
            "isolated_runtime_image": "sha256:" + "a" * 64,
            "output_token_budget": 8192,
            "native_timeout_seconds": 14400,
            "output_root": "out",
            "upstream_repo": "creator",
        },
    )
    with pytest.raises(ValueError, match="require isolated|mismatch: output_token_budget"):
        app.require_isolated_live(config, {})
    config.isolated_runtime_image = "sha256:" + "a" * 64
    config.output_token_budget = 8192
    app.require_isolated_live(config, {})
