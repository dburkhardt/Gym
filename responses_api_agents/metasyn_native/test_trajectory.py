import json
import pytest
from nemo_gym.config_types import ModelServerRef
from responses_api_agents.metasyn_native.trajectory import observations


def saved(directory, index, usage=True):
    (directory / f"{index:06d}.request.json").write_text(
        json.dumps({"messages": [{"role": "user", "content": f"actual prompt {index}"}]})
    )
    body = {
        "id": f"response-{index}",
        "choices": [{"message": {"role": "assistant", "content": f"actual response {index}"}}],
    }
    if usage:
        body["usage"] = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    (directory / f"{index:06d}.response.json").write_text(
        json.dumps({"status": 200, "body": json.dumps(body), "started_at": 1.0})
    )


def test_solver_and_judges_retain_actual_records_and_usage(tmp_path):
    for i in range(5):
        saved(tmp_path, i)
    trajectory, usage = observations(
        tmp_path,
        task_id="task",
        rollout_id="0-0",
        model_ref=ModelServerRef(type="responses_api_models", name="policy_model"),
    )
    assert [t.invocation_id for t in trajectory.turns] == ["solver"] + ["judge"] * 4
    assert trajectory.turns[2].question[0]["content"] == "actual prompt 2"
    assert trajectory.turns[2].model_calls[0].response_id == "response-2"
    assert usage.input_tokens == 15 and usage.output_tokens == 10
    saved(tmp_path, 2, False)
    assert (
        observations(
            tmp_path,
            task_id="task",
            rollout_id="0-0",
            model_ref=ModelServerRef(type="responses_api_models", name="policy_model"),
        )[1]
        is None
    )


def test_unresolved_call_does_not_become_complete_trajectory(tmp_path):
    saved(tmp_path, 0)
    (tmp_path / "000000.response.json").write_text(json.dumps({"status": 500, "body": "{}", "started_at": 1.0}))
    with pytest.raises(ValueError, match="Unresolved"):
        observations(
            tmp_path,
            task_id="task",
            rollout_id="0-0",
            model_ref=ModelServerRef(type="responses_api_models", name="policy_model"),
        )
