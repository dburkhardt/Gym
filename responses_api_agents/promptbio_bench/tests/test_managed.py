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
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert stopped.is_set()
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
@pytest.mark.parametrize("finish,expected", [("stop", "completed"), ("length", "model_budget_exhausted")])
async def test_native_solver_real_uds_guard(tmp_path, monkeypatch, finish, expected):
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
            pass

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
    assert json.loads((run / "solver-lifecycle.json").read_text())["cleanup_confirmed"]
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
