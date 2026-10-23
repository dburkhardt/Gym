# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from responses_api_agents.harbor_agent.measurement_review import apply_measurement_review


def test_actual_nested_evaluator_errors_exclude_numeric_reward(tmp_path: Path) -> None:
    # Exact saved response: HTTP200, nested missing-file errors, aggregate -1.
    response = json.loads((Path(__file__).parent / "fixtures/evaluator-missing-input.json").read_text())
    state = tmp_path / "artifacts/state"
    state.mkdir(parents=True)
    actual = (Path(__file__).parent / "fixtures/evaluator-best-score.json").read_text()
    assert json.loads(actual)["best_raw_scores"] == response["raw_scores"]
    (state / "best_score.json").write_text(actual)
    original = {"verifier_result": {"rewards": {"reward": -1.0}}}
    revised, kind, reason = apply_measurement_review(
        original,
        trial_dir=tmp_path,
        instance_id="naturebench::s42256-024-00916-5",
        import_path="benchmarks.naturebench.measurement_review:review",
    )
    assert kind == "harbor:scientific_measurement_invalid"
    assert "/problem/data/drd2_qed/x_test.csv" in reason
    assert revised["verifier_result"] == original["verifier_result"]
    assert "measurement_review" not in original


def test_later_valid_best_is_not_invalidated_by_prior_error(tmp_path: Path) -> None:
    state = tmp_path / "artifacts/state"
    state.mkdir(parents=True)
    fixture = json.loads((Path(__file__).parent / "fixtures/evaluator-best-score.json").read_text())
    fixture["best_attempt"] = 2
    fixture["best_raw_scores"] = {
        group: {key: 0.0 for key in metrics if key != "error"} for group, metrics in fixture["best_raw_scores"].items()
    }
    (state / "best_score.json").write_text(json.dumps(fixture))
    (state / "submissions.jsonl").write_text(json.dumps({"attempt": 1, "raw_scores": {"error": "old failure"}}))
    revised, kind, reason = apply_measurement_review(
        {"verifier_result": {"rewards": {"reward": -1.0}}},
        trial_dir=tmp_path,
        instance_id="naturebench::s42256-024-00916-5",
        import_path="benchmarks.naturebench.measurement_review:review",
    )
    assert kind is None and reason is None
    assert revised["measurement_review"]["metrics"]["reward"]["status"] == "valid"


@pytest.mark.parametrize(
    "raw", ["malformed", None, {"drd2_qed": {"SR@0.4": None}}, {"drd2_qed": {"SR@0.4": float("nan")}}]
)
def test_malformed_measurements_are_unresolved(tmp_path: Path, raw: object) -> None:
    state = tmp_path / "artifacts/state"
    state.mkdir(parents=True)
    (state / "best_score.json").write_text(
        "{" if raw == "malformed" else json.dumps({"finalized": True, "best_raw_scores": raw})
    )
    revised, kind, _ = apply_measurement_review(
        {"verifier_result": {"rewards": {"reward": -1.0}}},
        trial_dir=tmp_path,
        instance_id="naturebench::s42256-024-00916-5",
        import_path="benchmarks.naturebench.measurement_review:review",
    )
    assert kind == "harbor:scientific_measurement_invalid"
    assert revised["measurement_review"]["metrics"]["reward"]["status"] == "unresolved"
