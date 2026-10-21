from types import SimpleNamespace

import pytest

from benchmarks.chemiq.observation import digest
from resources_servers.campaign_creator_scoring import app


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid,budget", [(False, False), (True, False), (True, True)])
async def test_observation_projection(monkeypatch, invalid, budget):
    raw = {
        "uuid": "synthetic",
        "verifier_metadata": {"answer": "CCO"},
        "responses_create_params": {"input": [{"role": "user", "content": "Synthetic scoring control"}]},
        "response": {
            "id": "synthetic",
            "created_at": 0,
            "model": "offline-fixture",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        },
    }
    raw.update(failure_kind=None, failure_reason=None, mask_sample=False)
    raw["response"].update(
        status="incomplete" if budget else "completed",
        incomplete_details={"reason": "max_output_tokens"} if budget else None,
        error=None,
    )
    task = {"uuid": raw["uuid"], "verifier_metadata": raw["verifier_metadata"]}
    details = {
        "reward": 0.0,
        "creator_result": {"is_correct": False},
        "dependency_observation": {
            "task_sha256": digest(task),
            "observer_sha256": "fixture",
            "events": [{"kind": "fixture", "dependency_failure": invalid}],
        },
    }

    async def score(**kwargs):
        return details

    monkeypatch.setattr(app, "run_score", score)
    server = SimpleNamespace(config=SimpleNamespace(benchmark="chemiq", upstream_repo="unused", scoring_timeout_s=600))
    result = await app.CreatorResourcesServer.verify(server, app.VerifyRequest.model_validate(raw))
    assert result.reward == 0 and result.scoring_details == details
    assert result.mask_sample is invalid
    assert result.failure_kind == ("model_budget" if budget else "chemiq:dependency_failure" if invalid else None)
    assert result.metadata["verifier_result"]["rewards"]["creator_check_answer"] == 0

    from nemo_gym.result_disposition import summarize_dispositions

    summary = summarize_dispositions([result.model_dump(mode="json")])
    assert (
        not invalid or result.metadata["measurement_review"]["metrics"]["creator_check_answer"]["status"] == "invalid"
    )
    if not invalid:
        assert "measurement_review" not in result.metadata
    assert summary["counts"]["raw_scored_records"] == 1
    assert summary["counts"]["quality_selected_records"] == int(not invalid)
