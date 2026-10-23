# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib

import pytest

from benchmarks.bioagentbench import measurement_review
from nemo_gym.result_disposition import summarize_dispositions


@pytest.mark.parametrize("reward", [0.0, 1.0])
@pytest.mark.parametrize("known", [True, False])
def test_deseq_raw_score_retained_but_not_scientific_quality(tmp_path, monkeypatch, reward, known):
    source = tmp_path / "tasksets/bioagent/bioagent_taskset/scoring.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"scorer fixture")
    if known:
        monkeypatch.setattr(measurement_review, "DESEQ_SCORER_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    fields = measurement_review.result_fields("deseq", reward, tmp_path)
    record = {"reward": reward, **fields}
    report = summarize_dispositions([record], expected_count=1)
    assert record["reward"] == reward
    assert record["metadata"]["verifier_result"]["rewards"]["reward"] == reward
    assert record["metadata"]["measurement_review"]["evidence"]["audited_source_matches"] is known
    assert report["counts"]["raw_scored_records"] == 1
    assert report["counts"]["completed_result_records"] == 1
    assert report["counts"]["quality_selected_records"] == 0


def test_other_workflows_unchanged(tmp_path):
    assert measurement_review.result_fields("cystic-fibrosis", 0.0, tmp_path) == {}
