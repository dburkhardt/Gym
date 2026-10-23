# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import pytest

from benchmarks.ainsteinbench.measurement_review import TASK
from nemo_gym.result_disposition import summarize_dispositions
from responses_api_agents.harbor_agent.app import HarborVerifyResponse
from responses_api_agents.harbor_agent.measurement_review import apply_measurement_review


@pytest.mark.parametrize("reward", [0.0, 100.0])
@pytest.mark.parametrize(
    "task",
    [TASK, "ainsteinbench::ET_EinsteinEvolve_GRHydro_GRHydro_Source_cc", "ainsteinbench::MSB_pyscf_pyscf_pr2373"],
)
def test_native_serialized_result_retains_raw_score_with_exact_task_qualification(tmp_path, reward, task):
    trial = {"verifier_result": {"rewards": {"score": reward}}}
    original = copy.deepcopy(trial)
    reviewed, kind, reason = apply_measurement_review(
        trial,
        trial_dir=tmp_path,
        instance_id=task,
        import_path="benchmarks.ainsteinbench.measurement_review:review",
    )
    record = HarborVerifyResponse.model_validate(
        {
            "responses_create_params": {"input": []},
            "reward": reward,
            "instance_id": task,
            "metadata": reviewed,
            "mask_sample": reason is not None,
            "failure_kind": kind,
            "failure_reason": reason,
            "response": {
                "id": "control",
                "object": "response",
                "created_at": 0,
                "model": "control",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
        }
    ).model_dump(mode="json")
    assert trial == original
    assert record["reward"] == reward
    assert record["metadata"]["verifier_result"] == original["verifier_result"]
    report = summarize_dispositions([record], expected_count=1)
    assert report["counts"]["raw_scored_records"] == 1
    assert report["counts"]["completed_result_records"] == 1
    assert report["counts"]["quality_selected_records"] == (0 if task == TASK else 1)
    assert bool(record["metadata"]["measurement_review"]["metrics"]) == (task == TASK)
