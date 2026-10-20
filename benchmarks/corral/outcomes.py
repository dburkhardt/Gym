# SPDX-License-Identifier: Apache-2.0
"""Keep attempted-task outcomes separate from creator score availability."""


class NativeAgentIterationLimitError(RuntimeError):
    """Creator interaction budget ended without a submitted artifact."""


class NativeAgentSurrenderError(RuntimeError):
    """Creator surrender is attempted but has no native scorer measurement."""


def outcome(result: dict, runtime: dict) -> dict:
    if runtime.get("commit_hash") != result.get("commit_hash"):
        raise ValueError("Corral outcome evidence refers to a different commit")
    metadata = runtime.get("runtime_metadata", {})
    submitted = result.get("status") == "submitted"
    budget = (
        not submitted
        and metadata.get("agent_status") == "iteration_limit"
        and metadata.get("error_type") == "AgentOutcomeError"
        and result.get("error") == "agent exhausted its 50 interaction budget"
    )
    return {
        "kind": "submitted"
        if submitted
        else "model_budget_exhausted"
        if budget
        else "model_surrendered"
        if result.get("status") == "surrendered"
        else "native_failure_unclassified",
        "attempted": 1,
        "completed": int(submitted),
        "task_completion_failure": int(not submitted),
        "creator_score_available": False,
        "creator_score": None,
        "runtime_error_type": metadata.get("error_type"),
        "native_status": result.get("status"),
        "native_error": result.get("error"),
    }
