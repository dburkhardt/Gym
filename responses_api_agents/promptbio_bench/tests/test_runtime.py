# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from responses_api_agents.promptbio_bench.runtime import parse_score, safe_task_id, stage_inputs


def test_stage_hides_reference(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "data").mkdir()
    (task / "data/input.txt").write_text("input")
    (task / "ref_answer").mkdir()
    (task / "ref_answer/answer.txt").write_text("SECRET")
    (task / "eval.json").write_text("SECRET")
    (task / "task.json").write_text(json.dumps({"id": "a-1-1", "input_files": ["data/input.txt"]}))
    stage_inputs(task, tmp_path / "workspace")
    paths = {str(p.relative_to(tmp_path / "workspace")) for p in (tmp_path / "workspace").rglob("*")}
    assert paths == {"task.json", "data", "data/input.txt", "outputs"}
    assert (tmp_path / "workspace/data/input.txt").read_text() == "input"


@pytest.mark.parametrize("path", ["../eval.json", "/etc/passwd", "ref_answer/answer.txt", "data/../../etc/passwd"])
def test_unsafe_input_rejected(tmp_path, path):
    (tmp_path / "task.json").write_text(json.dumps({"input_files": [path]}))
    with pytest.raises(ValueError):
        stage_inputs(tmp_path, tmp_path / "workspace")


@pytest.mark.parametrize("score", [1.0, 0.0, 0.375])
def test_upstream_score_preserved(tmp_path, score):
    path = tmp_path / "score.json"
    path.write_text(json.dumps({"avg_similarity": score}))
    assert parse_score(path) == score


@pytest.mark.parametrize("score", [None, True, "1", -0.1, 1.1, float("nan")])
def test_invalid_grading_not_zero(tmp_path, score):
    path = tmp_path / "score.json"
    path.write_text(json.dumps({"avg_similarity": score}))
    with pytest.raises(ValueError):
        parse_score(path)


def test_missing_result(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_score(tmp_path / "missing.json")


def test_bad_id():
    with pytest.raises(ValueError):
        safe_task_id("../../secret")


def test_budget_reservation_enforced_and_settled(tmp_path):
    from responses_api_agents.promptbio_bench.budget_proxy import Budget

    meter = Budget(tmp_path / "ledger.json", 0.5)
    payload = {"messages": [{"role": "user", "content": "test"}]}
    index = meter.reserve(payload)
    with pytest.raises(ValueError, match="Unreconciled"):
        meter.reserve(payload)
    meter.settle(index, {"prompt_tokens": 12, "completion_tokens": 10})
    assert meter.entries[0]["cost_usd"] == pytest.approx(0.00018)
    assert payload["max_completion_tokens"] == 16384
    meter.reserve(payload)
    restored = Budget(tmp_path / "ledger.json", 0.5)
    with pytest.raises(ValueError, match="Unreconciled"):
        restored.reserve(payload)


def test_budget_fails_closed(tmp_path):
    from responses_api_agents.promptbio_bench.budget_proxy import Budget

    meter = Budget(tmp_path / "ledger.json", 0.001)
    with pytest.raises(ValueError, match="exhausted"):
        meter.reserve({"messages": []})
    assert not meter.entries


def test_solver_uses_workspace_owner_and_resumes(tmp_path, monkeypatch):
    import os

    from responses_api_agents.promptbio_bench import runtime

    task = tmp_path / "a-1-1"
    task.mkdir()
    (task / "task.json").write_text(json.dumps({"id": "a-1-1", "input_files": [], "timeout_seconds": 30}))
    commands = []

    def fake_command(argv, **kwargs):
        commands.append(argv)
        return "container"

    monkeypatch.setattr(runtime, "command", fake_command)
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(
        runtime, "completion", lambda *a, **kw: {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
    )
    kwargs = {
        "model": "example-model",
        "base_url": "https://example.invalid",
        "api_key": "test",
        "image": "sha256:test",
    }
    run = runtime.solve(task, tmp_path / "results", **kwargs)
    docker = commands[0]
    assert docker[docker.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert docker[docker.index("--network") + 1] == "none"
    assert str(task) not in " ".join(docker)
    assert runtime.solve(task, tmp_path / "results", **kwargs) == run
    assert len(commands) == 1


@pytest.mark.asyncio
async def test_gym_run_response_contract(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from responses_api_agents.promptbio_bench.app import PromptBioAgent, PromptBioRequest

    agent = PromptBioAgent.model_construct(
        config=SimpleNamespace(model_name="test-model", model_server=SimpleNamespace(name="policy_model")),
        server_client=SimpleNamespace(
            global_config_dict={
                "provider_dispatch_guard": True,
                "policy_model": {"responses_api_models": {"openai_model": {"host": "127.0.0.1", "port": 12345}}},
            }
        ),
    )

    async def evaluate(self, task_id, endpoint, rollout_id):
        assert endpoint == "http://127.0.0.1:12345/ng-rollout/test-rollout/v1"
        (tmp_path / "receipt.json").write_text(json.dumps({"status": "completed"}))
        return 0.75, str(tmp_path)

    monkeypatch.setattr(PromptBioAgent, "_evaluate", evaluate)
    monkeypatch.setattr("responses_api_agents.promptbio_bench.app.current_rollout_id", lambda: "test-rollout")
    body = PromptBioRequest.model_validate(
        {"responses_create_params": {"input": "task"}, "verifier_metadata": {"task_id": "a-1-1"}}
    )
    result = await agent.run(None, body)
    assert result.reward == 0.75
    assert result.response.model == "test-model"
    assert json.loads(result.response.output[0].content[0].text) == {
        "artifact_dir": str(tmp_path),
        "scoring": "upstream avg_similarity",
        "solver_disposition": "completed",
    }


def test_completion_never_resends_ambiguous_provider_failure(monkeypatch):
    import urllib.error

    from responses_api_agents.promptbio_bench.runtime import completion

    failures = [
        TimeoutError("ambiguous"),
        urllib.error.URLError("disconnected"),
        urllib.error.HTTPError("http://local", 503, "unavailable", {}, None),
        urllib.error.HTTPError("http://local", 429, "limited", {}, None),
    ]
    for failure in failures:
        calls = []

        def fail(request, **kwargs):
            calls.append(request)
            raise failure

        monkeypatch.setattr("urllib.request.urlopen", fail)
        try:
            completion("http://local/v1", "placeholder", {"model": "test"}, timeout=1)
        except type(failure):
            pass
        else:
            raise AssertionError("Expected original provider failure")
        assert len(calls) == 1
