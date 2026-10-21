# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from types import SimpleNamespace

import pytest

from responses_api_agents.bioagent_native.importer import MODEL, Agent, RunRequest


@pytest.mark.asyncio
async def test_saved_import_preserves_lineage_and_qualifies_time_window(tmp_path):
    trace = {
        "id": "saved-trace",
        "task": {"data": {"task_id": "task-a", "prompt": "Original prompt"}},
        "is_completed": True,
        "errors": [],
        "timing": {"start": 1, "scoring": {"end": 4}},
        "rewards": {"results_match": 0},
        "metrics": {"results_match": 0},
        "info": {"assessment": "Original assessment"},
    }
    trace_path = tmp_path / "traces.jsonl"
    provider_path = tmp_path / "providers.jsonl"
    trace_path.write_text(json.dumps(trace) + "\n")
    # An overlapping record is not proven to belong to this task.
    provider_path.write_text(json.dumps({"time": 2, "response": {"model": MODEL}, "task_id": "other-task"}) + "\n")
    agent = Agent.model_construct(config=SimpleNamespace(trace_path=trace_path, provider_trace=provider_path))
    body = RunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Original prompt"}]},
            "verifier_metadata": {"id": "task-a"},
            "task_id": "task-a",
            "_ng_task_index": 9,
            "_ng_rollout_index": 2,
            "agent_ref": {"name": "bioagent_importer"},
        }
    )
    result = (await agent.run(None, body)).model_dump()
    assert result["task_id"] == "task-a" and result["_ng_task_index"] == 9 and result["_ng_rollout_index"] == 2
    assert result["agent_ref"] == {"name": "bioagent_importer"}
    assert result["reward"] == 0 and result["imported_saved_execution"] and result["synthetic_summary_response"]
    metrics = result["native_metrics"]
    assert "actual_model_calls" not in metrics
    assert metrics["provider_records_in_time_window"] == 1
    assert metrics["provider_attribution"] == "time_window_only_not_task_verified"
    assert metrics["source_trace_sha256"] == hashlib.sha256(trace_path.read_bytes()).hexdigest()
    assert metrics["source_provider_sha256"] == hashlib.sha256(provider_path.read_bytes()).hexdigest()
    assert metrics["source_trace_id"] == "saved-trace"
