# SPDX-License-Identifier: Apache-2.0
import json

from nemo_gym.config_types import ModelServerRef
from responses_api_agents.moosechem_native.trajectory import observations


def test_real_stage_records_and_unresolved_intent(tmp_path):
    rows = []
    for n, stage in enumerate(("inspiration_screening", "hypothesis_generation", "evaluate")):
        rows.append(
            {
                "request_id": str(n),
                "stage": stage,
                "started": 20 + n,
                "request": {"messages": [{"role": "user", "content": stage}]},
                "response": {
                    "id": str(n),
                    "choices": [{"message": {"content": stage}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                },
            }
        )
    (tmp_path / "model-traces.jsonl").write_text("\n".join(map(json.dumps, rows)))
    ref = ModelServerRef(type="responses_api_models", name="policy_model")
    t, u = observations(tmp_path, task_id="task", rollout_id="rollout", model_ref=ref)
    assert [i.invocation_id for i in t.invocations] == [r["stage"] for r in rows]
    assert [turn.timestamp for turn in t.turns] == [20, 21, 22]
    assert [turn.model_calls[0].response_id for turn in t.turns] == ["0", "1", "2"]
    assert u.total_tokens == 15
    (tmp_path / "dispatch-intents.jsonl").write_text(json.dumps({"request_id": "pending", "stage": "evaluate"}))
    t, u = observations(tmp_path, task_id="task", rollout_id="rollout", model_ref=ref)
    assert u is None and t.invocations[-1].status == "incomplete"
    assert t.gaps[0].code == "unresolved_native_model_intent"


def test_failed_call_is_not_synthetic_turn(tmp_path):
    (tmp_path / "model-traces.jsonl").write_text(
        json.dumps({"request_id": "x", "stage": "evaluate", "error_type": "TimeoutError"})
    )
    t, u = observations(tmp_path, task_id="task", rollout_id="rollout", model_ref=None)
    assert not t.turns and u is None and t.invocations[0].status == "failed"
