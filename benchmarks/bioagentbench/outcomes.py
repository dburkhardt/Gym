# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retain upstream-scored failures and distinguish unscored attempted trajectories."""

BUDGET_STOPS = {
    "max_turns",
    "max_input_tokens",
    "max_output_tokens",
    "max_total_tokens",
    "context_length",
    "harness_timeout",
}


def disposition(trace: dict) -> dict:
    """Never manufacture a reward when the native scorer did not return one."""
    sampled = [node for node in trace.get("nodes", []) if node.get("sampled")]
    budget = trace.get("stop_condition") in BUDGET_STOPS or bool(
        sampled and sampled[-1].get("finish_reason") == "length"
    )
    reward = trace.get("rewards", {}).get("results_match")
    scored = (
        isinstance(reward, (int, float))
        and not isinstance(reward, bool)
        and reward in (0, 1)
        and trace.get("is_completed") is True
        and not trace.get("errors")
        and bool(trace.get("timing", {}).get("scoring", {}).get("end"))
    )
    metrics = trace.get("info", {}).get("gym_giab_benchmark_metrics")
    unresolved = isinstance(metrics, dict) and metrics.get("status") == "hap.py_failed"
    if unresolved:
        scored = False
    return {
        "native_benchmark_metrics": metrics,
        "count_as_attempted": True,
        "scientific_reward": reward if scored else None,
        "measurement_available": scored,
        "model_or_agent_budget_event": budget,
        "native_stop_condition": trace.get("stop_condition"),
        "native_errors": trace.get("errors", []),
        "raw_native_reward": reward,
        "disposition": "verifier_execution_failure_cause_unresolved"
        if unresolved
        else ("upstream_scored_budget_outcome" if budget else "upstream_scored_outcome")
        if scored
        else ("unscored_model_or_agent_budget_outcome" if budget else "unscored_native_failure_requires_review"),
    }
