# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from nemo_gym.rollout_collection import is_terminal_failure
from responses_api_agents.promptbio_bench.guarded_scorer import ValidatedJudgeTransport
from responses_api_agents.promptbio_bench.nemotron_proxy import MODEL, JudgeOutputBudgetExhausted, validate_result
from responses_api_agents.promptbio_bench.outcomes import JudgeBudgetFailure


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("fixture", ["b27", "b43"])
def test_saved_judge_budget_terminal_outcome(tmp_path, fixture):
    saved = json.loads((Path(__file__).parent / "fixtures" / f"{fixture}-judge-length.json").read_text())
    grade, run = tmp_path / "grade", tmp_path / "solver"
    grade.mkdir()
    body = json.dumps(saved["response"])

    class Guard(httpx.BaseTransport):
        count = 0

        def handle_request(self, request):
            self.count += 1
            return httpx.Response(200, content=body)

    guard = Guard()
    with httpx.Client(transport=ValidatedJudgeTransport(guard, grade / "judge-failure.json")) as client:
        with pytest.raises(JudgeOutputBudgetExhausted):
            client.post("http://local/chat/completions")
        with pytest.raises(RuntimeError, match="Prior judge failure"):
            client.post("http://local/chat/completions")
    assert guard.count == 1
    # Numeric ordering remains correct after capture ordinals grow beyond padding.
    write(grade / "calls/9.response.json", {"status": 200, "body": "earlier response"})
    write(grade / "calls/10.response.json", {"status": 200, "body": body})
    from responses_api_agents.promptbio_bench import guarded_scorer
    from responses_api_agents.promptbio_bench.runtime import SCORER_REVISION

    write(
        grade / "runtime-source.json",
        {
            "scorer_revision": SCORER_REVISION,
            "implementation_sha256": hashlib.sha256(Path(guarded_scorer.__file__).read_bytes()).hexdigest(),
        },
    )
    write(grade / "judge-lifecycle.json", {"cleanup_confirmed": True})
    write(run / "solver-lifecycle.json", {"cleanup_confirmed": True})
    write(run / "receipt.json", {"status": "completed"})
    write(run / "workspace/outputs/result.json", {"actual": "saved solver artifact"})
    write(grade / "execution.json", {"return_code": 1, "error_type": None, "stderr": "original traceback retained"})
    error = JudgeBudgetFailure(grade, [run])
    record = error.record({"verifier_metadata": {"task_id": fixture}, "reward": 0}, "stable-rollout")
    assert is_terminal_failure(record)
    assert record["_ng_failure_class"] == "judge_failed"
    assert record["creator_score"] is None and record["scientific_score"] is None
    assert "reward" not in record and "response" not in record
    assert record["rollout_id"] == "stable-rollout"
    assert (
        record["artifact_sha256"][str(run / "workspace/outputs/result.json")]
        == hashlib.sha256((run / "workspace/outputs/result.json").read_bytes()).hexdigest()
    )
    assert json.loads((grade / "execution.json").read_text())["stderr"] == "original traceback retained"
    assert json.loads((grade / "terminal-outcome.json").read_text()) == record
    for code in (0, 124, 137, 143, -9, None):
        write(grade / "execution.json", {"return_code": code, "error_type": None})
        with pytest.raises(ValueError, match="settled creator failure"):
            error.record({}, "stable-rollout")
    write(grade / "execution.json", {"return_code": 1, "error_type": "timeout"})
    with pytest.raises(ValueError, match="settled creator failure"):
        error.record({}, "stable-rollout")
    write(grade / "execution.json", {"return_code": 1, "error_type": None})
    source_path = grade / "runtime-source.json"
    source = json.loads(source_path.read_text())
    write(source_path, {**source, "implementation_sha256": "wrong"})
    with pytest.raises(ValueError, match="source identity mismatch"):
        error.record({}, "stable-rollout")
    write(source_path, source)
    receipt_path = grade / "judge-failure.json"
    receipt = json.loads(receipt_path.read_text())
    write(receipt_path, {**receipt, "response_sha256": "wrong"})
    with pytest.raises(ValueError, match="response identity mismatch"):
        error.record({}, "stable-rollout")
    write(receipt_path, receipt)
    for capture_path in (grade / "calls").glob("*.response.json"):
        capture_path.unlink()
    with pytest.raises(ValueError, match="lacks captured response"):
        error.record({}, "stable-rollout")
    write(grade / "judge-lifecycle.json", {"cleanup_confirmed": False})
    with pytest.raises(RuntimeError, match="cleanup unconfirmed"):
        error.record({}, "stable-rollout")


@pytest.mark.parametrize(
    "response",
    [
        {"model": MODEL, "choices": []},
        {"model": "wrong", "choices": [{"finish_reason": "length"}]},
        {"model": MODEL, "choices": [{"finish_reason": "error"}]},
    ],
)
def test_other_failures_are_not_judge_budget(response):
    with pytest.raises(ValueError) as caught:
        validate_result(response)
    assert not isinstance(caught.value, JudgeOutputBudgetExhausted)


@pytest.mark.asyncio
async def test_terminal_judge_record_collector_persists_and_does_not_replay(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    import nemo_gym.rollout_collection as collector
    from nemo_gym.reward_profile import compute_aggregate_metrics
    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper, _failures_path_for
    from tests.unit_tests.test_rollout_collection import FakeResponse, install_fake_server_client

    monkeypatch.setattr(collector, "get_global_config_dict", lambda: {})
    inputs, outputs = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    inputs.write_text(
        "".join(
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my_agent"}, "x": i}) + "\n"
            for i in range(2)
        )
    )
    calls, aggregate = [], []

    async def post(server_name, url_path, json, **kwargs):
        if url_path == "/run":
            calls.append(json["x"])
            if json["x"] == 0:
                return FakeResponse(
                    200,
                    {
                        "_ng_failure_class": "judge_failed",
                        "_ng_failure_terminal": True,
                        "creator_score": None,
                        "scientific_score": None,
                        "failure_disposition": "judge_output_budget_exhausted",
                        "solver_artifact_dirs": ["retained"],
                    },
                )
            return FakeResponse(200, {"reward": 1.0, "response": {"usage": {"total_tokens": 3}}})
        aggregate.extend(json.verify_responses)
        return FakeResponse(200, compute_aggregate_metrics(json.verify_responses).model_dump())

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(inputs),
        output_jsonl_fpath=str(outputs),
        disable_health_check=True,
        require_complete=False,
    )
    await RolloutCollectionHelper().run_from_config(config)
    failures = [json.loads(line) for line in _failures_path_for(outputs).read_text().splitlines()]
    assert len(failures) == 1 and is_terminal_failure(failures[0])
    assert "reward" not in failures[0] and failures[0]["creator_score"] is None
    assert len(outputs.read_text().splitlines()) == 1
    assert len(aggregate) == 1 and aggregate[0]["reward"] == 1.0
    before = list(calls)
    config.resume_from_cache = True
    await RolloutCollectionHelper().run_from_config(config)
    assert calls == before


@pytest.mark.asyncio
async def test_http_run_preserves_typed_failure_without_response_model_score(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fastapi import FastAPI

    from responses_api_agents.promptbio_bench import app

    agent = app.PromptBioAgent.model_construct(
        config=SimpleNamespace(model_server=SimpleNamespace(name="policy_model")),
        server_client=SimpleNamespace(
            global_config_dict={
                "provider_dispatch_guard": True,
                "policy_model": {"responses_api_models": {"openai_model": {"host": "127.0.0.1", "port": 12345}}},
            }
        ),
    )

    async def evaluate(self, task_id, endpoint, rollout_id):
        raise JudgeBudgetFailure(tmp_path, [])

    def record(self, body, rollout_id):
        return {**body, "_ng_failure_class": "judge_failed", "_ng_failure_terminal": True, "creator_score": None}

    monkeypatch.setattr(app.PromptBioAgent, "_evaluate", evaluate)
    monkeypatch.setattr(JudgeBudgetFailure, "record", record)
    monkeypatch.setattr(app, "current_rollout_id", lambda: "stable-rollout")
    web = FastAPI()
    web.post("/run")(agent.run)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web), base_url="http://local") as client:
        response = await client.post(
            "/run", json={"responses_create_params": {"input": "task"}, "verifier_metadata": {"task_id": "b-2-7"}}
        )
    assert response.status_code == 200
    assert response.json()["creator_score"] is None
    assert response.json()["_ng_failure_terminal"] is True
    assert "reward" not in response.json() and "response" not in response.json()
