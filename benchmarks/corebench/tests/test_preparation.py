# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

from benchmarks.corebench.prepare import CREATOR_IMAGE, TASK, stage, validate
from nemo_gym.eval_plan import PlanBenchmark, PlanGym, build_eval_plan


@pytest.fixture
def assets():
    root = Path(os.environ.get("COREBENCH_ASSET_ROOT", "../../outputs/broad-eval-campaign"))
    prepared = root / "harbor/agent-expansion/core-harbor"
    audit = root / "foundation/corebench-native/source-audit.json"
    runtime = root / "foundation/corebench-native/corebench-python311-runtime.tar.gz"
    if not prepared.exists() or not runtime.exists():
        pytest.skip("Pinned CORE-Bench campaign assets not staged")
    return prepared, audit, runtime


def test_native_preparation_preserves_references_outside_environment(assets, tmp_path):
    prepared, audit, runtime = assets
    output = stage(prepared, audit, runtime, tmp_path, docker_image="sha256:" + "a" * 64)
    row = json.loads(output.read_text())
    assert row["instance_id"] == row["task_id"] == f"corebench::{TASK}"
    plan = build_eval_plan(
        [row],
        benchmark=PlanBenchmark(name="corebench", config_paths=[]),
        taskset="corebench",
        gym=PlanGym(version="test", commit=None),
    )
    assert len(plan.tasks) == 1
    task = tmp_path / "tasks" / TASK
    assert (task / "tests/evaluations.py").read_bytes() == (
        prepared / "tasks" / TASK / "tests/evaluations.py"
    ).read_bytes()
    assert (task / "task.toml").read_text().startswith('artifacts = ["/workspace"]')
    assert 'docker_image = "sha256:' + "a" * 64 + '"' in (task / "task.toml").read_text()
    assert (task / "environment/python/bin/python3").exists()
    assert (task / "environment/Dockerfile").read_text().startswith("FROM " + CREATOR_IMAGE)
    assert not list((task / "environment").rglob("answers.json"))
    assert not (task / "environment/capsule/results").exists()
    assert json.loads((tmp_path / "provenance.json").read_text())["rows"] == 1


def test_changed_prepared_source_refused_before_publication(assets, tmp_path):
    prepared, audit, runtime = assets
    copy = tmp_path / "prepared"
    shutil.copytree(prepared, copy)
    (copy / "tasks" / TASK / "instruction.md").write_text("changed")
    with pytest.raises(ValueError, match="Changed prepared source"):
        stage(copy, audit, runtime, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_creator_scorer_correct_incorrect_and_missing(assets):
    prepared, audit, runtime = assets
    validate(prepared, audit, runtime)
    source = prepared / "tasks" / TASK / "tests"
    spec = importlib.util.spec_from_file_location("corebench_creator_evaluations", source / "evaluations.py")
    module = importlib.util.module_from_spec(spec)
    exec(compile((source / "evaluations.py").read_bytes(), str(source / "evaluations.py"), "exec"), module.__dict__)
    answers = json.loads((source / "answers.json").read_text())
    for response, expected in [(dict(answers[0]), 1), ({next(iter(answers[0])): "wrong answer"}, 0), ({}, 0)]:
        result = module.eval_result_json(answers, response)
        assert result["correct_written_answers"] + result["correct_vision_answers"] == expected
        assert result["total_written_questions"] + result["total_vision_questions"] == 1


def test_oracle_controls_keep_reference_outside_build_context(assets, tmp_path):
    from benchmarks.corebench.controls import stage_control

    prepared, _, _ = assets
    for correct in (True, False):
        destination = tmp_path / str(correct)
        stage_control(prepared / "tasks", destination, correct=correct)
        script = (destination / "solution/solve.sh").read_text()
        assert "/workspace/report.json" in script
        assert "test ! -e /tests/answers.json" in script
        assert "reference isolation passed" in script
        assert not list((destination / "environment").rglob("answers.json"))
        assert not list((destination / "environment").rglob("solve.sh"))
        assert ("deliberately incorrect control" in script) is (not correct)
        with pytest.raises(ValueError, match="overwrite"):
            stage_control(prepared / "tasks", destination, correct=correct)


def test_mutable_final_image_refused_before_staging(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        stage(tmp_path, tmp_path, tmp_path, tmp_path, docker_image="example:latest")
