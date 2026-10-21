# SPDX-License-Identifier: Apache-2.0
import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from nemo_gym.rollout_health import evaluate_record, run_health_checks
from nemo_gym.rollout_observability import AgentInvocation


def record():
    ref = {"model_ref": {"type": "responses_api_models", "name": "model"}, "response_id": "r1"}
    call = {
        "model_call_id": "c1",
        "request": {
            "model": "any-model",
            "prompt_logprobs": 0,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "benign"}],
        },
        "response": {
            "id": "r1",
            "model": "any-model",
            "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            "prompt_logprobs": [None, {"7": {"logprob": -1.0}}],
        },
        "response_metadata": {
            "model_ref": ref["model_ref"],
            "response_id": "r1",
            "model": "any-model",
            "status_code": 200,
            "finish_reason": "length",
        },
        "token_stats": {"prompt_tokens": 2, "completion_tokens": 1},
    }
    return {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "response": {"usage": {"input_tokens": 2, "output_tokens": 1}},
        "ng_trajectory": {
            "task_id": "task",
            "rollout_id": "0-0",
            "invocations": [
                {
                    "invocation_id": "likelihood",
                    "status": "completed",
                    "output_semantics": "prompt_likelihood",
                    "model_calls": [ref],
                }
            ],
            "turns": [
                {
                    "invocation_id": "likelihood",
                    "task_id": "task",
                    "rollout_id": "0-0",
                    "turn_no": 1,
                    "step_count": 1,
                    "timestamp": 1.0,
                    "answer": "",
                    "model_calls": [ref],
                }
            ],
            "model_calls": [call],
        },
    }


def checks(r):
    return {finding.check for finding in evaluate_record(r, frozenset()).findings}


def test_typed_semantics_and_valid_capture():
    assert AgentInvocation(invocation_id="legacy").output_semantics == "generation"
    with pytest.raises(ValidationError):
        AgentInvocation(invocation_id="x", output_semantics="ignore_empty")
    assert checks(record()) == set()


def test_generative_empty_length_stays_flagged():
    r = record()
    r["ng_trajectory"]["invocations"][0].pop("output_semantics")
    assert {"agent_turn_hollow", "model_call_runaway_generation"} <= checks(r)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "cardinality",
        "first_token",
        "positive",
        "nan",
        "multiple",
        "model",
        "metadata_model",
        "response_id",
        "model_ref",
        "missing_usage",
        "missing_capture",
        "request_mode",
        "incomplete",
    ],
)
def test_invalid_likelihood_never_exempts(mutation):
    r = record()
    t = r["ng_trajectory"]
    c = t["model_calls"][0]
    response = c["response"]
    if mutation == "missing":
        response.pop("prompt_logprobs")
    elif mutation == "cardinality":
        response["usage"]["prompt_tokens"] = 3
    elif mutation == "first_token":
        response["prompt_logprobs"][0] = {"3": {"logprob": -1}}
    elif mutation == "positive":
        response["prompt_logprobs"][1]["7"]["logprob"] = 1
    elif mutation == "nan":
        response["prompt_logprobs"][1]["7"]["logprob"] = float("nan")
    elif mutation == "multiple":
        response["prompt_logprobs"][1]["8"] = {"logprob": -2}
    elif mutation == "model":
        response["model"] = "wrong"
    elif mutation == "metadata_model":
        c["response_metadata"]["model"] = "wrong"
    elif mutation == "response_id":
        response["id"] = "other"
    elif mutation == "model_ref":
        c["response_metadata"]["model_ref"] = {"type": "responses_api_models", "name": "other"}
    elif mutation == "missing_usage":
        response.pop("usage")
    elif mutation == "missing_capture":
        t["model_calls"] = []
    elif mutation == "request_mode":
        c["request"]["max_tokens"] = 100
    elif mutation == "incomplete":
        t["invocations"][0]["status"] = "incomplete"
    found = checks(r)
    assert "model_call_invalid_likelihood_evidence" in found
    assert "agent_turn_hollow" in found


def test_invalid_likelihood_visible_even_with_nonempty_generation():
    r = record()
    r["ng_trajectory"]["turns"][0]["answer"] = "some text"
    r["ng_trajectory"]["model_calls"][0]["response"].pop("prompt_logprobs")
    assert "model_call_invalid_likelihood_evidence" in checks(r)


def test_other_health_checks_remain_active():
    r = record()
    r["response"]["usage"]["input_tokens"] = 99
    assert "rollout_token_count_mismatch" in checks(r)


def test_mixed_invocations_do_not_exempt_unrelated_generation():
    r = record()
    t = r["ng_trajectory"]
    second = deepcopy(t["model_calls"][0])
    second["model_call_id"] = "c2"
    second["response_metadata"]["response_id"] = "r2"
    second["response"]["id"] = "r2"
    t["model_calls"].append(second)
    reference = {"model_ref": second["response_metadata"]["model_ref"], "response_id": "r2"}
    t["invocations"].append({"invocation_id": "generation", "status": "completed", "model_calls": [reference]})
    turn = deepcopy(t["turns"][0])
    turn.update(invocation_id="generation", turn_no=2, step_count=2, model_calls=[reference])
    t["turns"].append(turn)
    found = checks(r)
    assert {"agent_turn_hollow", "model_call_runaway_generation"} <= found


@pytest.mark.parametrize(
    "gap",
    [
        "model_calls_unavailable",
        "model_call_capture_incomplete",
        "model_call_capture_records_unreadable",
        "model_call_capture_unreadable",
        "trajectory_projection_failed",
    ],
)
def test_inline_likelihood_claim_with_capture_gap_does_not_exempt(gap):
    r = record()
    r["ng_trajectory"]["gaps"] = [{"code": gap}]
    found = checks(r)
    if gap != "trajectory_projection_failed":
        assert "agent_turn_hollow" in found
        assert "model_call_invalid_likelihood_evidence" in found
    else:
        assert evaluate_record(r, frozenset()).verdict != "healthy"


def test_no_owned_calls_cannot_vacuously_qualify():
    r = record()
    r["ng_trajectory"]["invocations"][0]["model_calls"] = []
    assert {"agent_turn_hollow", "model_call_invalid_likelihood_evidence"} <= checks(r)


def test_full_reader_without_capture_projection_rejects_inline_claim(tmp_path):
    r = record()
    r["response"]["claimed_likelihood_receipt"] = r["ng_trajectory"]["model_calls"].pop()
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(r) + "\n")
    result = run_health_checks(path, workers=1)
    findings = {x.check for x in result.rollouts[0].findings}
    assert {"model_call_invalid_likelihood_evidence", "agent_turn_hollow"} <= findings
    assert result.rollouts[0].verdict != "healthy"


def test_same_call_claimed_by_generative_and_likelihood_invocations_is_not_exempt():
    r = record()
    t = r["ng_trajectory"]
    t["invocations"].append(
        {"invocation_id": "generative", "model_calls": deepcopy(t["invocations"][0]["model_calls"])}
    )
    assert {"agent_turn_hollow", "model_call_runaway_generation", "model_call_invalid_likelihood_evidence"} <= checks(
        r
    )
