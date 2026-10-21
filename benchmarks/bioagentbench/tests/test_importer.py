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
        "nodes": [{"message": {"role": "assistant", "content": "Unsuccessful attempt"}}],
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
    assert result["mask_sample"] is False
    metrics = result["native_metrics"]
    assert "actual_model_calls" not in metrics
    assert metrics["provider_records_in_time_window"] == 1
    assert metrics["provider_attribution"] == "time_window_only_not_task_verified"
    assert metrics["source_trace_sha256"] == hashlib.sha256(trace_path.read_bytes()).hexdigest()
    assert metrics["source_provider_sha256"] == hashlib.sha256(provider_path.read_bytes()).hexdigest()
    assert metrics["source_trace_id"] == "saved-trace"


@pytest.mark.asyncio
@pytest.mark.parametrize("nodes", [[], None, {}])
async def test_missing_execution_nodes_retains_raw_zero_but_masks(tmp_path, nodes):
    trace = {
        "id": "empty-trace",
        "task": {"data": {"task_id": "empty", "prompt": "Original"}},
        "is_completed": True,
        "errors": [],
        "nodes": nodes,
        "timing": {"start": 1, "scoring": {"end": 4}},
        "rewards": {"results_match": 0},
        "metrics": {},
        "info": {"assessment": "saved"},
    }
    result = await _import_fixture(tmp_path, trace)
    assert result.reward == 0 and result.mask_sample is True
    assert result.failure_kind == "historical_execution_unverified"
    assert result.native_metrics["rewards"] == {"results_match": 0}
    assert result.native_metrics["provider_attribution"] == "time_window_only_not_task_verified"


async def _import_fixture(tmp_path, trace):
    task = trace["task"]["data"]
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(trace) + "\n")
    before = path.read_bytes()
    provider = tmp_path / "provider.jsonl"
    provider.write_text(
        json.dumps({"time": trace["timing"]["start"], "response": {"model": MODEL}, "task_id": "unrelated"}) + "\n"
    )
    agent = Agent.model_construct(config=SimpleNamespace(trace_path=path, provider_trace=provider))
    body = RunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": task["prompt"]}]},
            "verifier_metadata": {"id": task["task_id"]},
            "task_id": task["task_id"],
        }
    )
    result = await agent.run(None, body)
    assert path.read_bytes() == before
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "task_id, masked", [("alzheimer-mouse", True), ("transcript-quant", False), ("comparative-genomics", False)]
)
async def test_preserved_zero_outcomes_require_execution_evidence(tmp_path, task_id, masked):
    from pathlib import Path

    campaign = Path(__file__).resolve().parents[5] / "outputs/broad-eval-campaign"
    inventory = campaign / "foundation/bioagent-managed/native-cystic-live/historical-trace-inventory.json"
    if not inventory.exists():
        pytest.skip("Preserved campaign trace unavailable")
    entry = next(x for x in json.loads(inventory.read_text()) if x["task_id"] == task_id)
    path = campaign.parents[1] / entry["copies"][0]["path"]
    original = path.read_bytes()
    assert hashlib.sha256(original).hexdigest() == entry["copies"][0]["sha256"]
    trace = json.loads(original.splitlines()[0])
    assert trace["is_completed"] and not trace["errors"]
    assert bool(trace["nodes"]) is not masked
    result = await _import_fixture(tmp_path, trace)
    assert result.reward == trace["rewards"]["results_match"] == 0
    assert result.mask_sample is masked
    assert result.failure_kind == ("historical_execution_unverified" if masked else None)
    assert path.read_bytes() == original
