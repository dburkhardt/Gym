# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Orchestration tests; upstream metric correctness is tested separately."""

import json
from unittest.mock import AsyncMock

import pytest

from responses_api_agents.genotex import app


def response(output):
    return {
        "id": "response1",
        "created_at": 1.0,
        "model": "test-model",
        "object": "response",
        "output": output,
        "parallel_tool_calls": False,
        "tools": [],
        "tool_choice": "auto",
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {})
    paths = {
        name: tmp_path / "runs" / "shared" / name
        for name in ["public", "prediction", "reference", "grading", "workspace"]
    }
    for path in paths.values():
        path.mkdir(parents=True)
    monkeypatch.setattr(app, "stage", lambda *_: paths)
    container = AsyncMock()
    container.inspect.return_value = {"image_id": "fixture-image", "network_mode": "none"}
    container.shell.return_value = {"exit_code": 0, "output": "done"}
    monkeypatch.setattr(app, "Container", lambda **_: container)
    scorer = AsyncMock(return_value={"reward": 0.5, "raw_upstream": {"analysis": "fixture"}})
    monkeypatch.setattr(app, "score", scorer)
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())

    async def parsed(reply):
        return reply

    monkeypatch.setattr(app, "get_response_json", parsed)
    client = AsyncMock()
    config = app.GenoTEXConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="test",
        model_server={"type": "responses_api_models", "name": "policy_model"},
        data_root=str(tmp_path),
        work_root=str(tmp_path / "runs"),
    )
    agent = app.GenoTEXAgent.model_construct(config=config, server_client=client)
    body = app.GenoTEXRunRequest(
        responses_create_params={"input": [{"role": "user", "content": "task"}]},
        verifier_metadata={"trait": "Trait", "condition": None, "traits": ["Trait"], "task_id": "Trait__None"},
    )
    return agent, body, client, container, scorer


@pytest.mark.asyncio
async def test_tool_loop_saves_trajectory_and_scores(setup):
    agent, body, client, container, scorer = setup
    client.post.side_effect = [
        response(
            [
                {
                    "type": "function_call",
                    "name": "shell",
                    "arguments": json.dumps({"command": "echo done"}),
                    "call_id": "call1",
                    "id": "fc1",
                }
            ]
        ),
        response([]),
    ]
    result = await agent.run(body)
    assert result.reward == 0.5
    assert not result.mask_sample
    assert result.verifier_metadata["task_id"] == "Trait__None"
    container.shell.assert_awaited_once_with("echo done")
    container.stop.assert_awaited_once()
    scorer.assert_awaited_once()
    second = client.post.call_args_list[1].kwargs["json"]
    assert second["input"][-1]["type"] == "function_call_output"
    assert second["input"][-1]["call_id"] == "call1"


@pytest.mark.asyncio
async def test_model_failure_is_masked_infrastructure(setup):
    agent, body, client, container, scorer = setup
    client.post.side_effect = RuntimeError("endpoint unavailable")
    result = await agent.run(body)
    assert result.mask_sample
    assert result.reward == 0
    assert "endpoint unavailable" in result.failure_reason
    scorer.assert_not_awaited()
    container.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_step_budget_scores_partial_outputs(setup):
    agent, body, client, container, scorer = setup
    agent.config.max_steps = 1
    client.post.return_value = response(
        [{"type": "function_call", "name": "shell", "arguments": "invalid json", "call_id": "call1", "id": "fc1"}]
    )
    result = await agent.run(body)
    assert result.failure_reason == "max_steps_exhausted"
    assert not result.mask_sample
    container.shell.assert_not_awaited()
    scorer.assert_awaited_once()


def test_shell_tool_matches_gym_request_schema():
    parsed = app.NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "user", "content": "task"}], tools=[app.SHELL_TOOL]
    )
    assert parsed.tools[0]["strict"] is True


@pytest.mark.asyncio
async def test_container_shared_spec_and_scoped_cleanup(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from responses_api_agents.genotex import runtime

    seen = {}
    sandbox = AsyncMock()
    sandbox.serialize.return_value = {"sandbox_id": "owned-fixture"}
    sandbox.exec.return_value = SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None)

    def factory(config, spec):
        seen.update(config=config, spec=spec)
        return sandbox

    monkeypatch.setattr(runtime, "AsyncSandbox", factory)
    paths = {name: tmp_path / name for name in ("public", "prediction", "workspace")}
    for path in paths.values():
        path.mkdir()
    container = runtime.Container(image="fixture", paths=paths, cpus=4, memory_gb=16, ttl_s=100)
    await container.start()
    await container.stop()
    await container.stop()
    spec = seen["spec"]
    assert spec.resources.memory_mib == 16384 and spec.ttl_s == 280
    assert seen["config"]["docker"]["create"]["network"] == "none"
    assert all("reference" not in v and "grading" not in v for v in spec.provider_options["volumes"])
    assert spec.provider_options["run_args"] == ["--user", f"{runtime.os.getuid()}:{runtime.os.getgid()}"]
    assert "/output/.write-check" in sandbox.exec.call_args.args[0]
    sandbox.stop.assert_awaited_once()
    assert json.loads((tmp_path / "sandbox-lifecycle.json").read_text())["cleanup_confirmed"]


@pytest.mark.asyncio
async def test_unwritable_container_is_stopped_before_inference(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from responses_api_agents.genotex import runtime

    sandbox = AsyncMock()
    sandbox.serialize.return_value = {"sandbox_id": "owned"}
    sandbox.exec.return_value = SimpleNamespace(return_code=1, stdout="denied", stderr="", error_type=None)
    monkeypatch.setattr(runtime, "AsyncSandbox", lambda *a: sandbox)
    paths = {name: tmp_path / name for name in ("public", "prediction", "workspace")}
    for path in paths.values():
        path.mkdir()
    container = runtime.Container(image="fixture", paths=paths, cpus=4, memory_gb=16)
    with pytest.raises(RuntimeError, match="not writable"):
        await container.start()
    sandbox.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_token_limit_is_a_task_failure_not_infrastructure(setup):
    agent, body, client, container, scorer = setup
    reply = response([])
    reply.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    client.post.return_value = reply
    result = await agent.run(body)
    assert result.failure_reason == "model_response_incomplete:max_output_tokens"
    assert not result.mask_sample
    scorer.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_preserves_completed_tool_and_original_budget(setup):
    from pathlib import Path

    agent, body, client, container, scorer = setup
    client.post.side_effect = [
        response(
            [
                {
                    "type": "function_call",
                    "name": "shell",
                    "arguments": json.dumps({"command": "echo done"}),
                    "call_id": "call1",
                    "id": "fc1",
                }
            ]
        ),
        RuntimeError("transport lost"),
    ]
    failed = await agent.run(body)
    checkpoint = Path(failed.artifact_dir) / "checkpoint.json"
    state = json.loads(checkpoint.read_text())
    assert state["next_step"] == 1
    assert len(list(Path(failed.artifact_dir).glob("tool-*.json"))) == 1
    assert state["history"][-1]["call_id"] == "call1"
    old_elapsed = state["elapsed_seconds"]
    resolution = {
        "action_id": state["pending"]["action_id"],
        "outcome": "not_forwarded",
        "checkpoint": str(checkpoint),
    }
    resumed = body.model_copy(deep=True)
    resumed.verifier_metadata.update(resume_checkpoint=str(checkpoint), resume_resolution=resolution)
    client.post.side_effect = [response([])]
    result = await agent.run(resumed)
    assert not result.mask_sample
    assert result.elapsed_seconds >= old_elapsed
    container.shell.assert_awaited_once_with("echo done")
    assert json.loads(checkpoint.read_text()) == state  # original record remains immutable
    assert len(json.loads((Path(result.artifact_dir) / "trajectory.json").read_text())) == 3


@pytest.mark.asyncio
async def test_uncertain_model_request_cannot_silently_repeat(setup):
    from pathlib import Path

    agent, body, client, container, scorer = setup
    client.post.side_effect = RuntimeError("transport lost")
    failed = await agent.run(body)
    resumed = body.model_copy(deep=True)
    resumed.verifier_metadata["resume_checkpoint"] = str(Path(failed.artifact_dir) / "checkpoint.json")
    result = await agent.run(resumed)
    assert result.mask_sample
    assert "receipt review" in result.failure_reason
    assert client.post.await_count == 1


def test_checkpoint_refuses_uncertain_tool_and_exhausted_budget(tmp_path):
    from responses_api_agents.genotex.checkpoint import remaining_budget, restore_checkpoint, save_checkpoint

    paths = {}
    for name in ("public", "prediction", "reference", "grading", "workspace"):
        folder = tmp_path / name
        folder.mkdir()
        paths[name] = str(folder)
    state = {
        "task": {"trait": "T"},
        "paths": paths,
        "pending": {"kind": "tool", "action_id": "unknown"},
        "elapsed_seconds": 10,
        "next_step": 2,
    }
    checkpoint = tmp_path / "checkpoint.json"
    save_checkpoint(checkpoint, state)
    with pytest.raises(ValueError, match="Unresolved tool"):
        restore_checkpoint(checkpoint, work_root=tmp_path, task={"trait": "T"})
    with pytest.raises(ValueError, match="budget exhausted"):
        remaining_budget(state, timeout_s=10, max_steps=80)
    with pytest.raises(ValueError, match="budget exhausted"):
        remaining_budget(state, timeout_s=7200, max_steps=2)


@pytest.mark.asyncio
async def test_guarded_call_binds_action_and_rejects_changed_image_resume(setup, monkeypatch):
    from pathlib import Path

    agent, body, client, container, scorer = setup
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {"provider_dispatch_guard": True})
    client.post.return_value = response([])
    result = await agent.run(body)
    assert not result.mask_sample
    kwargs = client.post.call_args.kwargs
    assert kwargs["headers"]["x-nemo-gym-logical-call-id"].startswith("genotex:")
    assert kwargs["_max_num_tries"] == 1 and kwargs["_resend_after_disconnect"] is False
    state = json.loads((Path(result.artifact_dir) / "checkpoint.json").read_text())
    assert state["guard_config_sha256"] == kwargs["headers"]["x-nemo-gym-call-config-sha256"]
    resumed = body.model_copy(deep=True)
    resumed.verifier_metadata["resume_checkpoint"] = str(Path(result.artifact_dir) / "checkpoint.json")
    container.inspect.return_value = {"image_id": "changed-image"}
    client.reset_mock()
    scorer.reset_mock()
    refusal = await agent.run(resumed)
    assert refusal.mask_sample and "matching recorded runtime" in refusal.failure_reason
    client.post.assert_not_awaited()
    scorer.assert_not_awaited()


@pytest.mark.asyncio
async def test_guarded_unknown_cannot_be_cleared_by_caller_assertion(setup, monkeypatch):
    from pathlib import Path

    agent, body, client, container, scorer = setup
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {"provider_dispatch_guard": True})
    client.post.side_effect = RuntimeError("unknown provider outcome")
    result = await agent.run(body)
    checkpoint = Path(result.artifact_dir) / "checkpoint.json"
    state = json.loads(checkpoint.read_text())
    resumed = body.model_copy(deep=True)
    resumed.verifier_metadata.update(
        resume_checkpoint=str(checkpoint),
        resume_resolution={
            "action_id": state["pending"]["action_id"],
            "outcome": "not_forwarded",
            "checkpoint": str(checkpoint),
        },
    )
    client.reset_mock()
    refusal = await agent.run(resumed)
    assert refusal.mask_sample and "caller assertion is insufficient" in refusal.failure_reason
    client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_shared_command_timeout_reaps_real_scientific_child(tmp_path):
    import os
    import sys

    from responses_api_agents.genotex import runtime

    marker = tmp_path / "pid"
    script = f"import os,time;open({str(marker)!r},'w').write(str(os.getpid()));time.sleep(30)"
    with pytest.raises(TimeoutError, match="process group settled"):
        await runtime.command(sys.executable, "-c", script, timeout=1)
    pid = int(marker.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.asyncio
async def test_grader_timeout_is_failure_without_second_scoring_attempt(tmp_path, monkeypatch):
    from responses_api_agents.genotex import runtime

    calls = AsyncMock(side_effect=TimeoutError("deadline"))
    monkeypatch.setattr(runtime, "command", calls)
    with pytest.raises(RuntimeError, match="no automatic regrade"):
        await runtime.score(
            data_root=tmp_path,
            run_root=tmp_path,
            paths={k: tmp_path for k in ("grading", "prediction", "reference")},
            task={"trait": "fixture", "condition": None},
            python="python3",
        )
    calls.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["model", "params", "runtime", "none", "later_resume"])
async def test_settled_error_resume_binding_and_new_lineage(setup, monkeypatch, change):
    """Restore validation is tested separately; exercise real dispatch ordering."""
    from pathlib import Path

    agent, body, client, container, scorer = setup
    monkeypatch.setattr(app, "execution_identity", AsyncMock(return_value={"image": "same"}))
    root = Path(agent.config.work_root)
    old = root / "old-checkpoint.json"
    history = [{"role": "user", "content": "original"}]
    params = body.responses_create_params.model_dump(exclude_none=True)
    params.update(tools=[app.SHELL_TOOL], parallel_tool_calls=False, temperature=0)
    retained = {**params, "input": history, "model": "original-model"}
    state = {
        "task": dict(body.verifier_metadata),
        "paths": {
            name: str(root / "shared" / name) for name in ("public", "prediction", "workspace", "reference", "grading")
        },
        "history": history,
        "trajectory": [],
        "next_step": 2,
        "elapsed_seconds": 10,
        "pending": None,
        "pending_calls": [],
        "solver_done": False,
        "runtime_identity": {"image": "same"},
        "settled_model_errors": [{"action_id": "failed-old-action", "request": retained}],
    }
    if change == "runtime":
        state["runtime_identity"] = {"image": "changed"}
    if change == "later_resume":
        # A historical settlement must not constrain an unrelated later resume.
        state["history"] = history + [{"role": "assistant", "content": "advanced"}]
    old.write_text(json.dumps(state))
    original = old.read_bytes()
    monkeypatch.setattr(app, "restore_checkpoint", lambda *a, **k: json.loads(old.read_text()))
    monkeypatch.setattr(
        app,
        "get_global_config_dict",
        lambda: {"policy_model_name": "changed" if change == "model" else "original-model"},
    )
    body.verifier_metadata["resume_checkpoint"] = str(old)
    if change != "later_resume":
        body.verifier_metadata["resume_resolution"] = {"outcome": "settled_error"}
    if change == "params":
        body.responses_create_params.max_output_tokens = 123
    dispatched_ids = []

    async def observed_dispatch(*args, **kwargs):
        checkpoints = [p for p in root.glob("*/checkpoint.json") if p != old]
        live = json.loads(checkpoints[-1].read_text())
        dispatched_ids.append(live["pending"]["action_id"])
        return response([])

    client.post.side_effect = observed_dispatch
    result = await agent.run(body)
    assert old.read_bytes() == original
    if change in ("model", "params", "runtime"):
        assert result.mask_sample
        client.post.assert_not_awaited()
    else:
        assert not result.mask_sample
        client.post.assert_awaited_once()
        assert len(dispatched_ids) == 1 and dispatched_ids[0] != "failed-old-action"
        derived = json.loads((Path(result.artifact_dir) / "checkpoint.json").read_text())
        assert derived["settled_model_errors"][0]["action_id"] == "failed-old-action"
        assert Path(result.artifact_dir) != old.parent
        assert derived["next_step"] == 3
        assert derived["elapsed_seconds"] >= 10
    container.shell.assert_not_awaited()
