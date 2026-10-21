# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from benchmarks.bioagentbench.outcomes import disposition


def test_upstream_scored_timeout_zero_is_valid_model_outcome():
    result = disposition(
        {
            "is_completed": True,
            "timing": {"scoring": {"end": 1}},
            "stop_condition": "harness_timeout",
            "rewards": {"results_match": 0.0},
            "errors": [],
        }
    )
    assert result["measurement_available"] and result["scientific_reward"] == 0
    assert result["model_or_agent_budget_event"] and result["count_as_attempted"]


def test_partial_reward_with_scoring_error_is_not_valid():
    result = disposition(
        {
            "is_completed": True,
            "rewards": {"results_match": 0.0},
            "timing": {"scoring": {"end": None}},
            "errors": [{"type": "TaskError", "message": "scoring failed"}],
        }
    )
    assert not result["measurement_available"] and result["scientific_reward"] is None
    assert result["raw_native_reward"] == 0


def test_unscored_length_retains_attempt_without_inventing_zero():
    result = disposition(
        {"is_completed": True, "rewards": {}, "nodes": [{"sampled": True, "finish_reason": "length"}]}
    )
    assert result["scientific_reward"] is None and result["count_as_attempted"]
    assert result["disposition"] == "unscored_model_or_agent_budget_outcome"


def test_unknown_native_error_is_not_assumed_infrastructure():
    result = disposition({"is_completed": True, "rewards": {}, "errors": [{"type": "TaskError"}]})
    assert result["scientific_reward"] is None
    assert result["disposition"] == "unscored_native_failure_requires_review"
