# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify two frozen reference mismatches without replacing creator scores."""

import hashlib
import json
from pathlib import Path


AUDITED_SOURCES = {
    "t_test--v6__dfeep__paper_155__task_7": {
        "tests/evaluate.py": "26f161f9528eaa875c8752ca68ed8cb73391d99d09f80adc60e8ce42ece60d21",
        "tests/answer_key.json": "def99552e9da77af0545c9456d547f6ed9101fc2e19ff1ee860c4455bb5c06c3",
        "instruction.md": "04e828dcdf8afb31326f36f6fdf586751e956b12a66a58d93c0abc16bb289cca",
        "environment/dataset.csv": "ad8873003a8733dec1579806d79e0a22abbe4a905e7e81f2805a4d0568b8101a",
    },
    "t_test--v6__dfeep__paper_193__task_2": {
        "tests/evaluate.py": "26f161f9528eaa875c8752ca68ed8cb73391d99d09f80adc60e8ce42ece60d21",
        "tests/answer_key.json": "dbe4a9c001811208f4e1cf00b648ea03fb2b28dc999dab114b6844342a3a6f25",
        "instruction.md": "58afaa9d5c722af06826e5373992d81cadee15f0f60a8b7fbc5b358a030365bf",
        "environment/dataset.csv": "8826ac98a592047bff0696e5c1f5496f820f9d1bb14c61a7cd9f32b3ad0ea1ae",
    },
}


def review(*, trial_dir: Path, instance_id: str, trial_result: dict) -> dict:
    """Retain raw scores and exclude unresolved audited references from quality means."""
    result = {"version": "pbench-v6-reference-review-v1", "metrics": {}, "evidence": {}}
    task_id = instance_id.removeprefix("pbench::")
    if task_id not in AUDITED_SOURCES:
        return result
    score_path = trial_dir / "verifier/managed-score/score.json"
    score = json.loads(score_path.read_text())
    native = (trial_result.get("verifier_result") or {}).get("rewards") or {}
    known = score.get("task_id") == task_id and score.get("source_hashes") == AUDITED_SOURCES[task_id]
    consistent = all(native.get(k) == score.get("strict" if k == "reward" else k) for k in native)
    result["evidence"] = {
        "dataset": "pbench",
        "task_id": task_id,
        "score_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
        "source_hashes": score.get("source_hashes", {}),
    }
    reason = (
        "Pinned v6 reference is unresolved: independent two-sided log-outcome Welch and pooled t-tests do not reproduce the released p-value at the stated sample size. Creator reference-generation procedure is required; raw scores are retained, not replaced."
        if known and consistent
        else "Audited v6 task has changed or missing scorer/input identity or inconsistent raw values; a new reference review is required."
    )
    result["metrics"] = {
        name: {"status": "unresolved", "reason": reason} for name in native if name in ("reward", "raw", "strict")
    }
    return result
