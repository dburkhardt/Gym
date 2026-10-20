# SPDX-License-Identifier: Apache-2.0
import pytest

from benchmarks.corral.outcomes import outcome
from responses_api_agents.harbor_agent.failure_diagnostics import measurement_failure


@pytest.mark.parametrize(
    "error_type,agent_status,expected",
    [
        ("AgentOutcomeError", "iteration_limit", "model_budget_exhausted"),
        ("APIConnectionError", "failed", "native_failure_unclassified"),
        ("ToolExecutionError", "failed", "native_failure_unclassified"),
        ("RuntimeError", "failed", "native_failure_unclassified"),
    ],
)
def test_budget_requires_creator_terminal_metadata(error_type, agent_status, expected):
    result = {"commit_hash": "same", "status": "failed", "error": "agent exhausted its 50 interaction budget"}
    runtime = {"commit_hash": "same", "runtime_metadata": {"error_type": error_type, "agent_status": agent_status}}
    record = outcome(result, runtime)
    assert record["kind"] == expected
    assert record["attempted"] == 1 and record["task_completion_failure"] == 1
    assert record["creator_score"] is None and not record["creator_score_available"]


def test_named_budget_is_unscored_model_outcome_not_infrastructure():
    kind, _ = measurement_failure(
        {
            "reward_valid": False,
            "exception_info": {"exception_type": "NativeAgentIterationLimitError", "exception_message": "native50"},
        }
    )
    assert kind == "harbor:agent_outcome_unscored"
    kind, _ = measurement_failure({"reward_valid": False, "exception_info": {"exception_type": "APIConnectionError"}})
    assert kind != "harbor:agent_outcome_unscored"
