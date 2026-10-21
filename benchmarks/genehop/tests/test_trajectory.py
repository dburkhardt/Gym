# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest

from nemo_gym.config_types import ModelServerRef
from responses_api_agents.genehop_native.trajectory import observations


def test_native_receipts_join_without_fabricated_usage(tmp_path):
    raw = {
        "turn": 0,
        "started_at": 3,
        "request": {"messages": [{"role": "user", "content": "actual"}]},
        "response": {"id": "native-id", "choices": [{"message": {"content": "actual response"}}]},
    }
    (tmp_path / "model-00.json").write_text(json.dumps(raw))
    body = b"actual tool body"
    (tmp_path / "tool-001.body").write_bytes(body)
    (tmp_path / "tool-001.json").write_text(
        json.dumps(
            {
                "url": "https://example.invalid/fixture",
                "sha256": hashlib.sha256(body).hexdigest(),
                "time": 4,
                "started_at": 3,
            }
        )
    )
    t, usage = observations(
        tmp_path,
        task_id="task",
        rollout_id="rollout",
        model_ref=ModelServerRef(type="responses_api_models", name="policy_model"),
        status="step_budget",
    )
    assert usage is None
    assert t.turns[0].question == raw["request"]["messages"]
    assert t.turns[0].model_calls[0].response_id == "native-id"
    assert t.tool_calls[0].output == body.decode()
    assert t.invocations[0].stop_reason == "max_iterations"
    (tmp_path / "tool-001.body").write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        observations(tmp_path, task_id="task", rollout_id="rollout", model_ref=None, status="completed")
