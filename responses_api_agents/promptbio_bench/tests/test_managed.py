# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import os
from contextlib import asynccontextmanager

import httpx
import pytest

from responses_api_agents.promptbio_bench import managed_runtime as runtime
from responses_api_agents.promptbio_bench.guarded_scorer import ValidatedJudgeTransport


@pytest.mark.parametrize("image", [None, "latest", "sha256:short", "repo@sha256:" + "g" * 64])
def test_image_gate(image):
    with pytest.raises(ValueError):
        runtime.validate_image(image)


def test_saved_output_fifo_rejected_without_read(tmp_path):
    os.mkfifo(tmp_path / "result.json")
    with pytest.raises(ValueError, match="special file"):
        runtime.validate_saved_outputs(tmp_path)


def test_judge_failure_poison_prevents_resend():
    class Guard(httpx.BaseTransport):
        count = 0

        def handle_request(self, request):
            self.count += 1
            return httpx.Response(
                200,
                json={
                    "model": runtime.MODEL,
                    "choices": [{"finish_reason": "length", "message": {"content": "unfinished"}}],
                },
            )

    guard = Guard()
    with httpx.Client(transport=ValidatedJudgeTransport(guard)) as client:
        with pytest.raises(Exception):
            client.post("http://judge/v1/chat/completions")
        with pytest.raises(RuntimeError, match="Prior judge failure"):
            client.post("http://judge/v1/chat/completions")
    assert guard.count == 1


@pytest.mark.asyncio
async def test_paired_judge_boundary_and_cancellation(tmp_path, monkeypatch):
    task = tmp_path / "task"
    task.mkdir()
    (task / "reference.txt").write_text("protected")
    creator = tmp_path / "creator"
    creator.mkdir()
    runs = []
    for name in ("first", "second"):
        run = tmp_path / name
        (run / "workspace/outputs").mkdir(parents=True)
        (run / "receipt.json").write_text("{}")
        runs.append(run)
    (runs[0] / "workspace/outputs/value.txt").write_text("frozen input")
    output = tmp_path / "grade"
    entered = asyncio.Event()
    stopped = asyncio.Event()
    specs = []

    class Sandbox:
        def __init__(self, config, spec):
            specs.append(spec)

        async def start(self):
            pass

        async def serialize(self):
            return {"owned": "test"}

        async def stop(self):
            await asyncio.sleep(0.01)
            stopped.set()

        async def exec(self, command, **kwargs):
            entered.set()
            await asyncio.Event().wait()

    @asynccontextmanager
    async def bridge(socket_path, **kwargs):
        yield

    async def valid_source(source):
        pass

    async def creator_snapshot(source, destination):
        snapshot = destination / "creator-source"
        snapshot.mkdir()
        return snapshot, {}

    monkeypatch.setattr(runtime, "snapshot_creator", creator_snapshot)
    monkeypatch.setattr(runtime, "validate_creator_source", valid_source)
    monkeypatch.setattr(runtime, "AsyncSandbox", Sandbox)
    monkeypatch.setattr(runtime, "bridge", bridge)
    pending = asyncio.create_task(
        runtime.grade_managed(
            task,
            runs,
            scorer_dir=creator,
            python="python",
            output=output,
            endpoint="http://local/ng-rollout/test/v1",
            rollout_id="test",
            scorer_image="sha256:" + "a" * 64,
        )
    )
    await asyncio.wait_for(entered.wait(), 2)
    (runs[0] / "workspace/outputs/value.txt").write_text("later host mutation")
    assert (output / "candidate-inputs/0/value.txt").read_text() == "frozen input"
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert stopped.is_set()
    signature = json.loads((output / "identity.json").read_text())["signature"]
    assert signature["task_files"]["reference.txt"]
    assert signature["implementation"]["guarded_chat_transport.py"]
    assert signature["controller_files"]
    import hashlib

    assert signature["candidate_files"] == [{"value.txt": hashlib.sha256(b"frozen input").hexdigest()}, {}]
    receipt = json.loads((output / "judge-lifecycle.json").read_text())
    assert receipt["cleanup_confirmed"]
    request = json.loads((output / "request.json").read_text())
    assert request["runs"] == ["/runs/0/first", "/runs/1/second"]
    mounts = specs[0].provider_options["volumes"]
    for target in (
        "/creator:ro",
        "/controller:ro",
        "/task:ro",
        "/results/controller:ro",
        "/results/task-source:ro",
        "/results/request.json:ro",
    ):
        assert any(m.endswith(":" + target) for m in mounts)
    assert sum("/workspace/outputs:ro" in m for m in mounts) == 2
    assert not any("inference" in str(v) for v in specs[0].env.values())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finish,expected,cleanup_failure",
    [("stop", "completed", False), ("length", "model_budget_exhausted", False), ("length", "failed", True)],
)
async def test_native_solver_real_uds_guard(tmp_path, monkeypatch, finish, expected, cleanup_failure):
    from nemo_gym.sandbox import model_bridge

    task = tmp_path / "a-1-1"
    task.mkdir()
    (task / "task.json").write_text(json.dumps({"id": "a-1-1", "input_files": [], "timeout_seconds": 30}))
    calls = []
    specs = []

    class Sandbox:
        def __init__(self, config, spec):
            specs.append(spec)

        async def start(self):
            pass

        async def serialize(self):
            return {"owned": "solver"}

        async def stop(self):
            if cleanup_failure:
                raise RuntimeError("Injected cleanup failure")

    class Reply:
        status = 200

        async def read(self):
            return json.dumps(
                {
                    "model": runtime.MODEL,
                    "choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": "done"}}],
                }
            ).encode()

        def release(self):
            pass

    async def upstream(method, url, **kwargs):
        calls.append((url, kwargs))
        return Reply()

    monkeypatch.setattr(runtime, "AsyncSandbox", Sandbox)
    monkeypatch.setattr(model_bridge, "request", upstream)
    arguments = dict(
        endpoint="http://local/ng-rollout/native-test/v1",
        rollout_id="native-test",
        model=runtime.MODEL,
        image="sha256:" + "a" * 64,
    )
    if cleanup_failure:
        with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
            await runtime.solve_managed(task, tmp_path / "runs", **arguments)
    else:
        await runtime.solve_managed(task, tmp_path / "runs", **arguments)
    run = next((tmp_path / "runs/a-1-1").iterdir())
    assert json.loads((run / "receipt.json").read_text())["status"] == expected
    identity = json.loads((run / "identity.json").read_text())
    assert identity["signature"]["task_json_sha256"]
    assert len(calls) == 1
    assert calls[0][0].endswith("/ng-rollout/native-test/v1/chat/completions")
    assert calls[0][1]["headers"]["x-nemo-gym-logical-call-id"].startswith(
        "promptbio_solver:" + identity["lineage"] + ":"
    )
    assert calls[0][1]["_max_num_tries"] == 1
    assert len(list((run / "calls").glob("*.request.json"))) == 1
    assert json.loads((run / "solver-lifecycle.json").read_text())["cleanup_confirmed"] is (not cleanup_failure)
    assert len(specs[0].provider_options["volumes"]) == 1
    assert not any("reference" in m or "model.sock" in m for m in specs[0].provider_options["volumes"])


def test_mapping_reference_self_compare_rejected(tmp_path):
    from types import SimpleNamespace

    from responses_api_agents.promptbio_bench.guarded_scorer import validate_mapping

    ref = str(tmp_path / "reference")
    candidate = str(tmp_path / "candidate")
    mapping = SimpleNamespace(mappings=[SimpleNamespace(reference_file=ref, candidate_file=candidate)])
    validate_mapping(mapping, reference_files=[ref], candidate_files=[candidate])
    mapping.mappings[0].candidate_file = ref
    with pytest.raises(ValueError, match="outside"):
        validate_mapping(mapping, reference_files=[ref], candidate_files=[candidate])
    mapping.mappings[0].candidate_file = None
    validate_mapping(mapping, reference_files=[ref], candidate_files=[candidate])
    mapping.mappings *= 2
    with pytest.raises(ValueError, match="duplicate"):
        validate_mapping(mapping, reference_files=[ref], candidate_files=[candidate])


@pytest.mark.asyncio
@pytest.mark.parametrize("infrastructure_failure", [False, True])
async def test_exhausted_outputs_reach_scorer_not_infrastructure(tmp_path, monkeypatch, infrastructure_failure):
    from types import SimpleNamespace

    from responses_api_agents.promptbio_bench import app

    called = []

    async def download(*args):
        return tmp_path

    async def solve(*args, **kwargs):
        if infrastructure_failure:
            raise ConnectionError("uncertain provider")
        (tmp_path / "receipt.json").write_text(json.dumps({"status": "model_budget_exhausted"}))
        return tmp_path

    async def grade(task, runs, **kwargs):
        assert json.loads((runs[0] / "receipt.json").read_text())["status"] == "model_budget_exhausted"
        called.append(runs)
        return [0.25]

    monkeypatch.setattr(app, "download_managed", download)
    monkeypatch.setattr(app, "solve_managed", solve)
    monkeypatch.setattr(app, "grade_managed", grade)
    config = SimpleNamespace(
        scorer_image="sha256:" + "a" * 64,
        sandbox_image="sha256:" + "b" * 64,
        cache_dir=tmp_path,
        output_dir=tmp_path,
        model_name=runtime.MODEL,
        comparison_models=[],
        max_steps=100,
        max_tokens=16384,
        scorer_dir=tmp_path,
        scorer_python="python",
    )
    agent = app.PromptBioAgent.model_construct(config=config)
    if infrastructure_failure:
        with pytest.raises(ConnectionError):
            await agent._evaluate("task", "http://local/ng-rollout/x/v1", "x")
        assert not called
    else:
        assert (await agent._evaluate("task", "http://local/ng-rollout/x/v1", "x"))[0] == 0.25
        assert len(called) == 1


@pytest.mark.asyncio
async def test_creator_source_rejects_ignored_file(tmp_path, monkeypatch):
    import subprocess

    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / ".gitignore").write_text("private.env\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", ".gitignore"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Control",
            "-c",
            "user.email=control@example.invalid",
            "commit",
            "-m",
            "control",
        ],
        check=True,
        capture_output=True,
    )
    revision = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(runtime, "SCORER_REVISION", revision)
    await runtime.validate_creator_source(tmp_path)
    projection = tmp_path.parent / (tmp_path.name + "-projection")
    projection.mkdir()
    snapshot, manifest = await runtime.snapshot_creator(tmp_path, projection)
    assert manifest and not (snapshot / ".git").exists()
    assert set(manifest) == {".gitignore"}
    (tmp_path / "private.env").write_text("CONTROL_ONLY=placeholder\n")
    with pytest.raises(ValueError, match="ignored"):
        await runtime.validate_creator_source(tmp_path)


def test_native_request_preserves_rollout_indices():
    from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
    from responses_api_agents.promptbio_bench.app import PromptBioRequest

    raw = {
        "responses_create_params": {"input": []},
        "verifier_metadata": {"task_id": "b-4-3"},
        "_ng_task_index": 4,
        "_ng_rollout_index": 2,
    }
    request = PromptBioRequest.model_validate(raw)
    assert request.model_dump()["_ng_task_index"] == 4
    assert request.model_dump()["_ng_rollout_index"] == 2
    assert maybe_rollout_id_from_run_body(request) == maybe_rollout_id_from_run_body(raw)
    assert maybe_rollout_id_from_run_body(request) is not None


def test_real_assistant_response_round_trip_preserves_science():
    from pydantic import TypeAdapter
    from nemo_gym.openai_utils import NeMoGymChatCompletionAssistantMessageParam

    # Exact response-only field shape observed in the first native b-4-3 call.
    message = {
        "content": "\n",
        "refusal": None,
        "role": "assistant",
        "annotations": None,
        "audio": None,
        "function_call": None,
        "tool_calls": [
            {
                "id": "call-6670ae7d-5f16-4e24-93e6-03c1cbaa3318",
                "function": {
                    "arguments": '{"command":"ls -la /workspace && ls -la /workspace/data 2>/dev/null && cat /workspace/task.json 2>/dev/null"}',
                    "name": "shell",
                },
                "type": "function",
            }
        ],
        "reasoning_content": "Let me start by examining the task and the data file.",
        "provider_specific_fields": {"refusal": None},
    }
    import copy

    original = copy.deepcopy(message)
    result = runtime.assistant_request_message(message)
    assert message == original
    assert result["content"] == message["content"]
    assert result["tool_calls"] == message["tool_calls"]
    assert result["reasoning_content"] == message["reasoning_content"]
    assert "annotations" not in result and "provider_specific_fields" not in result
    assert TypeAdapter(NeMoGymChatCompletionAssistantMessageParam).validate_python(result) == result


@pytest.mark.parametrize(
    "field,value",
    [
        ("annotations", [{"type": "citation", "text": "evidence"}]),
        ("provider_specific_fields", {"refusal": "cannot comply"}),
        ("unknown_reasoning", "scientific rationale"),
    ],
)
def test_unsupported_assistant_semantics_fail_closed(field, value):
    with pytest.raises(ValueError, match="Unsupported non-null assistant response field"):
        runtime.assistant_request_message({"role": "assistant", "content": "answer", field: value})


def test_supported_assistant_semantics_are_not_discarded():
    message = {
        "role": "assistant",
        "content": None,
        "refusal": "Cannot comply",
        "audio": {"id": "audio123"},
        "function_call": {"name": "legacy", "arguments": '{"x":1}'},
        "reasoning_content": "rationale",
    }
    assert runtime.assistant_request_message(message) == message
