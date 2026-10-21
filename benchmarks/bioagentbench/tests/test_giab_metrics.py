# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from benchmarks.bioagentbench.creator_execution import GiabMetricCapture
from benchmarks.bioagentbench.outcomes import disposition


def completed(metrics, reward=0.0):
    return {
        "is_completed": True,
        "timing": {"scoring": {"end": 1}},
        "rewards": {"results_match": reward},
        "errors": [],
        "info": {"gym_giab_benchmark_metrics": metrics},
    }


@pytest.mark.asyncio
async def test_saved_malformed_control_shape_is_preserved_not_attributed():
    # Exact native-controls-v1/malformed/receipt.json result, not an inferred cause.
    raw = {"status": "hap.py_failed", "f1_score": 0.0, "exit_code": 255}

    async def scorer(*args):
        return raw

    capture = GiabMetricCapture(scorer)
    assert await capture("giab", "query", "truth") is raw
    trace = completed(None)
    del trace["info"]["gym_giab_benchmark_metrics"]
    capture.attach([trace])
    raw["exit_code"] = 99
    outcome = disposition(trace)
    assert outcome["native_benchmark_metrics"]["exit_code"] == 255
    assert outcome["raw_native_reward"] == 0
    assert outcome["scientific_reward"] is None and not outcome["measurement_available"]
    assert outcome["disposition"] == "verifier_execution_failure_cause_unresolved"


@pytest.mark.parametrize("metrics", [{"f1_score": 0.0}, {"status": "missing_query_vcf", "f1_score": 0.0}])
def test_genuine_zero_and_missing_query_keep_creator_meaning(metrics):
    result = disposition(completed(metrics))
    assert result["measurement_available"] and result["scientific_reward"] == 0


def test_saved_positive_shape_remains_available():
    metrics = {
        "type": "SNP",
        "truth_total": 34432,
        "query_total": 35136,
        "recall": 1.0,
        "precision": 0.9999709999999999,
        "f1_score": 0.999985,
    }
    assert disposition(completed(metrics, 1.0))["scientific_reward"] == 1


@pytest.mark.asyncio
async def test_saved_empty_control_exception_is_unchanged():
    error = ValueError("could not convert string to float: ''")

    async def scorer(*args):
        raise error

    capture = GiabMetricCapture(scorer)
    with pytest.raises(ValueError) as caught:
        await capture("giab", "query", "truth")
    assert caught.value is error and capture.result is None
    traces = [{"info": {}}]
    capture.attach(traces)
    assert traces == [{"info": {}}]


@pytest.mark.asyncio
async def test_single_rollout_scope_and_reserved_observation():
    async def scorer(*args):
        return {"f1_score": 0.0}

    capture = GiabMetricCapture(scorer)
    with pytest.raises(ValueError):
        await capture("other", "query", "truth")
    await capture("giab", "query", "truth")
    with pytest.raises(ValueError):
        await capture("giab", "query", "truth")
    with pytest.raises(ValueError):
        capture.attach([{}, {}])
    with pytest.raises(ValueError):
        capture.attach([{"info": {"gym_giab_benchmark_metrics": {}}}])
