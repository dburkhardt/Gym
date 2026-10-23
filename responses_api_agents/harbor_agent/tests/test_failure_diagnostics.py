# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real saved-result regressions plus explicit policy mutations (not new runs)."""

import copy
import json
from pathlib import Path

import pytest

from responses_api_agents.harbor_agent.failure_diagnostics import measurement_failure


def receipt(name):
    return json.loads((Path(__file__).parent / "fixtures/measurement_validity" / f"{name}.json").read_text())[
        "projection"
    ]


def test_real_incorrect_zero_remains_valid():
    result = receipt("valid_zero")
    before = copy.deepcopy(result)
    assert measurement_failure(result) == (None, None)
    assert result == before


def test_real_verifier_failure_and_partial_score_remain_masked():
    result = receipt("verifier_missing")
    assert measurement_failure(result)[0] is not None
    result["verifier_result"] = {"rewards": {"reward": 1.0}}
    assert measurement_failure(result)[0] is not None
    assert result["verifier_result"]["rewards"]["reward"] == 1


def test_real_provider_admission_budget_is_not_agent_budget():
    assert measurement_failure(receipt("provider_admission_budget"))[0] == "harbor:unclassified"


@pytest.mark.parametrize("kind", ["AgentTimeoutError", "ContextLengthExceededError", "OutputTokenExceededError"])
def test_native_agent_budget_with_recorded_zero_is_valid(kind):
    result = receipt("valid_zero")
    result["exception_info"] = {"exception_type": kind}
    assert measurement_failure(result) == (None, None)
    result["verifier_result"] = None
    assert measurement_failure(result)[0] == "harbor:agent_outcome_unscored"


@pytest.mark.parametrize(
    "kind", ["BudgetExceededError", "APIConnectionError", "MemoryLimitExceededError", "UnknownError"]
)
def test_non_agent_errors_do_not_become_valid_when_partial_reward_exists(kind):
    result = receipt("valid_zero")
    result["exception_info"] = {"exception_type": kind}
    assert (
        measurement_failure(result)[0]
        == {
            "BudgetExceededError": "provider_quota_exhausted",
            "APIConnectionError": "transport_peer_drop",
            "MemoryLimitExceededError": "provider_oom_killed",
            "UnknownError": "harbor:unclassified",
        }[kind]
    )


def test_native_validity_and_nested_exception():
    result = receipt("valid_zero")
    result["step_results"] = [{"exception_info": {"exception_type": "VerifierTimeoutError"}}]
    assert measurement_failure(result)[0] is not None
    result["reward_valid"] = False
    result["failure_cause"] = "verifier_failed"
    assert measurement_failure(result)[0] is not None
    result["reward_valid"] = True
    result["step_results"] = []
    assert measurement_failure(result) == (None, None)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0", None])
def test_invalid_rewards_are_not_measurements(value):
    result = receipt("valid_zero")
    result["verifier_result"]["rewards"]["reward"] = value
    assert measurement_failure(result)[0] is not None


def test_unresolved_model_transport_overrides_partial_score():
    result = receipt("valid_zero")
    assert measurement_failure(result, llm_failure="transport completion unknown")[0] == "harbor:unclassified"


@pytest.mark.parametrize("name", ["ContextLengthExceededError", "OutputTokenExceededError"])
def test_llm_flag_for_native_budget_does_not_mask_scored_outcome(name):
    result = receipt("valid_zero")
    result["exception_info"] = {"exception_type": name}
    flag = {"exception_type": name, "message": "native limit", "retry_performed": False}
    assert measurement_failure(result, llm_failure=flag) == (None, None)


def test_cancelled_and_native_sandbox_causes_are_distinct():
    result = receipt("valid_zero")
    result.update(reward_valid=False, failure_cause="cancelled")
    assert measurement_failure(result)[0] == "cancelled"
    result["failure_cause"] = "sandbox_failed"
    assert measurement_failure(result)[0] == "provider_unavailable"


@pytest.mark.parametrize("manifest", [None, {}, [], [{"source": "/workspace", "status": "failed"}],
    [{"source": "/workspace", "status": "skipped"}],
    [{"source": "/workspace", "status": "ok"}] * 2])
def test_required_partial_artifacts_cannot_be_valid_measurements(manifest):
    from responses_api_agents.harbor_agent.failure_diagnostics import required_artifact_failure
    assert required_artifact_failure(manifest, ["/workspace"])[0] is not None


@pytest.mark.parametrize("status", ["ok", "empty"])
def test_required_empty_capture_is_not_itself_model_or_infrastructure_failure(status):
    from responses_api_agents.harbor_agent.failure_diagnostics import required_artifact_failure
    assert required_artifact_failure([{"source": "/workspace", "status": status}], ["/workspace"]) == (None, None)
    assert required_artifact_failure([{"source": "/optional-log", "status": "failed"}], []) == (None, None)


def test_missing_model_submission_is_not_infrastructure_failure():
    for native in ({}, {"reward_valid": False, "failure_cause": "verifier_failed"}):
        receipt = {**native, "exception_info": {"exception_type": "ModelMissingSubmissionError", "exception_message": "No submission"}, "verifier_result": None}
        kind, reason = measurement_failure(receipt)
        assert kind == "model_missing_submission"
        assert reason
        assert receipt["verifier_result"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "call",
        "config",
        "payload",
        "pending",
        "unknown",
        "later_success",
        "earlier_rejection",
        "other_intent",
        "other_error",
        "rollout",
        "malformed",
        "symlink",
    ],
)
def test_exact_native_intent_rejection_attribution(tmp_path, mutation):
    from responses_api_agents.harbor_agent.failure_diagnostics import intent_bound_provider_rejection

    native = tmp_path / "trial/agent/native"
    native.mkdir(parents=True)
    capture = tmp_path / "capture"
    capture.mkdir()
    endpoint = "http://127.0.0.1:123/ng-rollout/0-0/v1/chat/completions"
    result = {
        "exception_info": {
            "exception_type": "ClientResponseError",
            "exception_message": f"500, message='Internal Server Error', url='{endpoint}'",
        },
        "verifier_result": {"rewards": {"reward": 0}},
    }
    before = copy.deepcopy(result)
    intent = {
        "headers": {"x-nemo-gym-logical-call-id": "call1", "x-nemo-gym-call-config-sha256": "a" * 64},
        "payload": {"messages": [{"role": "user", "content": "unchanged"}]},
    }
    record = {
        "provider_binding": {"logical_call": "call1", "logical_rollout": "0-0", "agent_config": "a" * 64},
        "request": copy.deepcopy(intent["payload"]),
        "status_code": 429,
        "error_category": "rate_limit",
        "provider_attempts": [{"attempt_id": "attempt1", "status_code": 429, "outcome": "rejected"}],
    }
    records = [record]
    if mutation == "call":
        record["provider_binding"]["logical_call"] = "unrelated"
    if mutation == "config":
        record["provider_binding"]["agent_config"] = "different"
    if mutation == "payload":
        record["request"] = {"messages": []}
    if mutation == "pending":
        (capture / "0-0.test.capture.pending").write_text("{}")
    if mutation == "unknown":
        record["provider_attempts"][0]["outcome"] = "unknown"
    if mutation == "later_success":
        records.append({"provider_binding": {"logical_call": "later"}, "status_code": 200})
    if mutation == "earlier_rejection":
        records.insert(0, {"provider_binding": {"logical_call": "earlier"}, "status_code": 429})
    if mutation == "other_intent":
        (native / "call-2.intent.json").write_text(json.dumps(intent))
    if mutation == "other_error":
        result["exception_info"]["exception_type"] = "VerifierTimeoutError"
    if mutation == "rollout":
        record["provider_binding"]["logical_rollout"] = "other-rollout"
    (native / "call-0.intent.json").write_text(json.dumps(intent))
    if mutation == "malformed":
        (native / "call-0.intent.json").write_text("{broken")
    if mutation == "symlink":
        target = tmp_path / "external-intent.json"
        target.write_text(json.dumps(intent))
        (native / "call-0.intent.json").unlink()
        (native / "call-0.intent.json").symlink_to(target)
    (capture / "0-0.capture.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    answer = intent_bound_provider_rejection(
        result, trial_dir=tmp_path / "trial", capture_dir=capture, rollout_id="0-0", endpoint=endpoint
    )
    assert (answer[0] if answer else None) == ("provider_quota_exhausted" if mutation is None else None)
    assert result["verifier_result"] == before["verifier_result"]


def test_ainstein_iteration_limit_is_unscored_agent_outcome():
    # Projection of native-v7: creator loop exhausted30 iterations; verifier never ran.
    result = {
        "reward_valid": False,
        "failure_cause": "unclassified",
        "exception_info": {"exception_type": "NativeIterationLimitError",
                           "exception_message": "Agent did not produce a solution within iteration limit"},
        "verifier_result": None,
    }
    before = copy.deepcopy(result)
    assert measurement_failure(result)[0] == "harbor:agent_outcome_unscored"
    assert result == before
    assert measurement_failure(result, llm_failure={"exception_type": "APIConnectionError"})[0] == "transport_peer_drop"
