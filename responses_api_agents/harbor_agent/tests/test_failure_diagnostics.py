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
