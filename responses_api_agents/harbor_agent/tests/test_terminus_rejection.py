import copy
import hashlib
import json
import pytest
from responses_api_agents.harbor_agent.failure_diagnostics import terminus_bound_provider_rejection


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "request",
        "config",
        "call",
        "rollout",
        "pending",
        "unknown",
        "completed",
        "prior_pending",
        "duplicate",
        "endpoint",
        "exception",
        "extra",
        "symlink",
        "malformed",
    ],
)
def test_cursor_binding(tmp_path, mutation):
    agent = tmp_path / "trial/agent"
    agent.mkdir(parents=True)
    captures = tmp_path / "captures"
    captures.mkdir()
    endpoint = "http://localhost:123/ng-rollout/0-0/v1/chat/completions"
    result = {
        "exception_info": {
            "exception_type": "HTTPStatusError",
            "exception_message": f"Server error '500 Internal Server Error' for url '{endpoint}'\nFor more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500",
        }
    }
    state = {"version": 1, "lineage": "a" * 32, "calls": {}}
    records = []
    for i in range(2):
        payload = {"messages": [{"role": "user", "content": str(i)}]}
        call = f"harbor:{state['lineage']}:solver:{i}"
        state["calls"][str(i)] = {
            "logical_call_id": call,
            "configuration_sha256": "b" * 64,
            "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "completed": i == 0,
        }
        records.append(
            {
                "model_call_id": str(i),
                "provider_binding": {"logical_call": call, "logical_rollout": "0-0", "agent_config": "b" * 64},
                "request": payload,
                "status_code": 200 if i == 0 else 429,
                "error_category": None if i == 0 else "rate_limit",
                "provider_attempts": [
                    {
                        "attempt_id": str(i),
                        "status_code": 200 if i == 0 else 429,
                        "outcome": "captured" if i == 0 else "rejected",
                    }
                ],
            }
        )
    if mutation == "request":
        records[-1]["request"] = {}
    if mutation in ("config", "call", "rollout"):
        records[-1]["provider_binding"][
            {"config": "agent_config", "call": "logical_call", "rollout": "logical_rollout"}[mutation]
        ] = "other"
    if mutation == "pending":
        (captures / "0-0.test.capture.pending").write_text("{}")
    if mutation == "unknown":
        records[-1]["provider_attempts"][0]["outcome"] = "unknown"
    if mutation == "completed":
        state["calls"]["1"]["completed"] = True
    if mutation == "prior_pending":
        state["calls"]["0"]["completed"] = False
    if mutation == "duplicate":
        records[-1]["provider_attempts"][0]["attempt_id"] = "0"
    if mutation == "endpoint":
        endpoint += "/other"
    if mutation == "exception":
        result["exception_info"]["exception_type"] = "TimeoutError"
    if mutation == "extra":
        records.append(copy.deepcopy(records[-1]))
    path = agent / "gym-call-state.json"
    path.write_text(json.dumps(state))
    if mutation == "malformed":
        path.write_text("{}")
    if mutation == "symlink":
        path.rename(tmp_path / "external.json")
        path.symlink_to(tmp_path / "external.json")
    (captures / "0-0.capture.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    before = copy.deepcopy(result)
    answer = terminus_bound_provider_rejection(
        result, trial_dir=tmp_path / "trial", capture_dir=captures, rollout_id="0-0", endpoint=endpoint
    )
    assert (answer is not None) == (mutation is None)
    assert result == before
