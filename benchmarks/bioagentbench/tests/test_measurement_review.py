# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib

import pytest

from benchmarks.bioagentbench import measurement_review
from nemo_gym.result_disposition import summarize_dispositions
from responses_api_agents.bioagent_native.app import VerifyResponse


@pytest.mark.parametrize("task_id", ["deseq", "single-cell"])
@pytest.mark.parametrize("reward", [0.0, 1.0])
@pytest.mark.parametrize("known", [True, False])
def test_raw_score_retained_but_not_scientific_quality(tmp_path, monkeypatch, task_id, reward, known):
    source = tmp_path / "tasksets/bioagent/bioagent_taskset/scoring.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"scorer fixture")
    if known:
        monkeypatch.setattr(
            measurement_review,
            "DESEQ_SCORER_SHA256" if task_id == "deseq" else "SINGLE_CELL_SCORER_SHA256",
            hashlib.sha256(source.read_bytes()).hexdigest(),
        )
    fields = measurement_review.result_fields(task_id, reward, tmp_path)
    record = VerifyResponse.model_validate(
        {
            "responses_create_params": {"input": []},
            "verifier_metadata": {"id": task_id},
            "_ng_task_index": 7,
            "_ng_rollout_index": 0,
            "reward": reward,
            "native_metrics": {},
            "artifact_dir": "fixture",
            "response": {
                "id": "fixture",
                "object": "response",
                "created_at": 0,
                "model": "fixture",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
            **fields,
        }
    ).model_dump(mode="json")
    assert record["_ng_task_index"] == 7
    report = summarize_dispositions([record], expected_count=1)
    assert record["reward"] == reward
    assert record["metadata"]["verifier_result"]["rewards"]["reward"] == reward
    assert record["metadata"]["measurement_review"]["evidence"]["audited_source_matches"] is known
    if known and task_id == "single-cell":
        assert record["metadata"]["measurement_review"]["creator_metric"] == "any_cluster_cell_type_pair_overlap"
        assert "without genes or statistics" in record["failure_reason"]
    assert report["counts"]["raw_scored_records"] == 1
    assert report["counts"]["completed_result_records"] == 1
    assert report["counts"]["quality_selected_records"] == 0


def test_other_workflows_unchanged(tmp_path):
    assert measurement_review.result_fields("cystic-fibrosis", 0.0, tmp_path) == {}
