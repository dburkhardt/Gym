# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import ttest_ind

from benchmarks.pbench.measurement_review import AUDITED_SOURCES, review
from benchmarks.pbench.regrade import regrade
from responses_api_agents.harbor_agent.measurement_review import apply_measurement_review


SOURCE = Path(__file__).resolve().parents[5] / "outputs/broad-eval-campaign/foundation/pbench-native/prepared50"


@pytest.mark.parametrize(
    "suffix,outcome,treatment,n,expected",
    [
        ("155__task_7", "reached_goal", "highinterest", 2265, 0.0022557961995123806),
        ("193__task_2", "anyfert_plus2_05", "safi_lr04", 1230, 0.12246586473758071),
    ],
)
def test_actual_reference_and_preservation(tmp_path, suffix, outcome, treatment, n, expected):
    task = "t_test--v6__dfeep__paper_" + suffix
    if not SOURCE.exists():
        pytest.skip("Pinned campaign inputs unavailable")
    for relative, digest in AUDITED_SOURCES[task].items():
        assert hashlib.sha256((SOURCE / "tasks" / task / relative).read_bytes()).hexdigest() == digest
    rows = list(csv.DictReader((SOURCE / "tasks" / task / "environment/dataset.csv").open()))
    groups = [[np.log(float(r[outcome])) for r in rows if float(r[treatment]) == group] for group in (0, 1)]
    assert sum(map(len, groups)) == n
    p = float(ttest_ind(*groups, equal_var=False).pvalue)
    assert p == pytest.approx(expected)
    reference = json.loads((SOURCE / "tasks" / task / "tests/answer_key.json").read_text())
    assert abs(p - reference["p_value"]) > 0.1
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "result.json").write_text(
        json.dumps({"predicted": {"p_value": p, "conclusion": "Reject H0" if p < 0.05 else "Fail to reject H0"}})
    )
    score = regrade(SOURCE, task, workspace)
    dest = tmp_path / "verifier/managed-score"
    dest.mkdir(parents=True)
    (dest / "score.json").write_text(json.dumps(score))
    raw = {"verifier_result": {"rewards": {"reward": score["strict"], "strict": score["strict"], "raw": score["raw"]}}}
    revised, kind, _ = apply_measurement_review(
        raw,
        trial_dir=tmp_path,
        instance_id="pbench::" + task,
        import_path="benchmarks.pbench.measurement_review:review",
    )
    assert kind == "harbor:scientific_measurement_invalid"
    assert revised["verifier_result"] == raw["verifier_result"]
    assert all(m["status"] == "unresolved" for m in revised["measurement_review"]["metrics"].values())
    score["source_hashes"] = {}
    (dest / "score.json").write_text(json.dumps(score))
    changed = review(trial_dir=tmp_path, instance_id="pbench::" + task, trial_result=raw)
    assert "changed or missing" in changed["metrics"]["strict"]["reason"]


def test_other_tasks_unreviewed(tmp_path):
    assert review(trial_dir=tmp_path, instance_id="pbench::other", trial_result={})["metrics"] == {}


def test_missing_score_is_measurement_failure_not_zero(tmp_path):
    trial = {"verifier_result": {"rewards": {"reward": 0.75}}}
    revised, kind, reason = apply_measurement_review(
        trial,
        trial_dir=tmp_path,
        instance_id="pbench::t_test--v6__dfeep__paper_155__task_7",
        import_path="benchmarks.pbench.measurement_review:review",
    )
    assert kind == "harbor:measurement_review_error"
    assert "FileNotFoundError" in reason
    assert revised["verifier_result"] == trial["verifier_result"]
