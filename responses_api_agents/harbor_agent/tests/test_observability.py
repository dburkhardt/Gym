import copy
import hashlib
import json

import pytest

from responses_api_agents.harbor_agent.observability import completed_terminus_observations


def packet():
    request = {"messages": [{"role": "user", "content": "task"}]}
    call = {
        "completed": True,
        "logical_call_id": "harbor:" + "a" * 32 + ":solver:0",
        "configuration_sha256": "b" * 64,
        "request_sha256": hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
    }
    state = {"version": 1, "lineage": "a" * 32, "calls": {"0": call}}
    capture = {
        "model_call_id": "call-id",
        "status_code": 200,
        "response": {"id": "response"},
        "request": request,
        "provider_binding": {
            "logical_rollout": "0-0",
            "logical_call": call["logical_call_id"],
            "agent_config": call["configuration_sha256"],
        },
    }
    return state, [capture]


def test_explicit_ownership_does_not_invent_outcome():
    state, captures = packet()
    before = copy.deepcopy((state, captures))
    invocation = completed_terminus_observations(state, captures, rollout_id="0-0").records[0]
    assert invocation.model_calls[0].model_call_id == "call-id"
    assert invocation.status == "unknown" and invocation.conversation == []
    assert (state, captures) == before


@pytest.mark.parametrize(
    "mutation", ["incomplete", "rollout", "call", "config", "request", "rejected", "response", "extra"]
)
def test_unproven_bindings_rejected(mutation):
    state, captures = packet()
    c = captures[0]
    if mutation == "incomplete":
        state["calls"]["0"]["completed"] = False
    elif mutation == "rollout":
        c["provider_binding"]["logical_rollout"] = "judge"
    elif mutation == "call":
        c["provider_binding"]["logical_call"] = "judge"
    elif mutation == "config":
        c["provider_binding"]["agent_config"] = "c" * 64
    elif mutation == "request":
        c["request"]["messages"] = []
    elif mutation == "rejected":
        c["status_code"] = 429
    elif mutation == "response":
        c["response"] = None
    elif mutation == "extra":
        captures.append(copy.deepcopy(c))
    with pytest.raises(ValueError):
        completed_terminus_observations(state, captures, rollout_id="0-0")


def test_missing_cursor_is_observation_gap(tmp_path):
    from responses_api_agents.harbor_agent.observability import read_terminus_observations

    result = read_terminus_observations(trial_dir=tmp_path, capture_dir=tmp_path, rollout_id="0-0")
    assert result.records == []
    assert result.gaps[0].code == "harbor_solver_call_bindings_unavailable"


def test_symlink_cursor_is_not_trusted(tmp_path):
    from responses_api_agents.harbor_agent.observability import read_terminus_observations

    (tmp_path / "agent").mkdir()
    target = tmp_path / "other.json"
    target.write_text(json.dumps(packet()[0]))
    (tmp_path / "agent/gym-call-state.json").symlink_to(target)
    result = read_terminus_observations(trial_dir=tmp_path, capture_dir=tmp_path, rollout_id="0-0")
    assert not result.records and result.gaps
