# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from benchmarks.naturebench import evaluation_inputs as snapshots
from benchmarks.naturebench.packaging import MOLECULAR_TASK
from benchmarks.naturebench.snapshot_packaging import install_snapshot_variant, patch_evaluation


FIXTURE = Path(__file__).parent / "fixtures/creator-evaluation.py.txt"


def service(monkeypatch, tmp_path, evaluator):
    """Load the actual patched creator controller; scientific math is fixture-only."""
    package = "snapshot_control_fixture"
    monkeypatch.setitem(sys.modules, package, types.ModuleType(package))
    scoring = types.ModuleType(package + ".scoring")
    scoring.load_primary_table = lambda path: {}
    scoring.compute_improvements = lambda raw, table: ({}, 1.0)
    monkeypatch.setitem(sys.modules, scoring.__name__, scoring)
    timer = types.ModuleType(package + ".timer")

    class Timer:
        def begin_evaluation(self):
            pass

        def end_evaluation(self):
            pass

    timer.EffectiveTimer = Timer
    monkeypatch.setitem(sys.modules, timer.__name__, timer)
    monkeypatch.setitem(sys.modules, package + ".evaluation_inputs", snapshots)
    module = types.ModuleType(package + ".evaluation")
    module.__file__ = str(FIXTURE)
    exec(
        compile(patch_evaluation(FIXTURE.read_text(), capture_timeout_seconds=60), str(FIXTURE), "exec"),
        module.__dict__,
    )
    source = (tmp_path / "output").resolve()
    source.mkdir()
    (source / "submission.csv").write_text("original")
    return module.SingleTrialEvaluator(
        "fixture", tmp_path / "metadata", source, tmp_path / "state", evaluator=evaluator
    )


def test_concurrent_attempts_retain_original_after_live_mutation(monkeypatch, tmp_path):
    barrier = threading.Barrier(3)
    captured = []

    def evaluate(path):
        captured.append(path)
        barrier.wait(timeout=5)
        barrier.wait(timeout=5)
        assert (path / "submission.csv").read_text() == "original"
        return {"fixture": True}

    instance = service(monkeypatch, tmp_path, evaluate)
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(instance.evaluate) for _ in range(2)]
        barrier.wait(timeout=5)
        (instance.output_dir / "submission.csv").write_text("mutated")
        barrier.wait(timeout=5)
        results = [future.result() for future in futures]
    assert {r["attempt"] for r in results} == {1, 2}
    assert len(set(captured)) == 2
    records = [json.loads(x) for x in instance.submissions_path.read_text().splitlines()]
    for record in records:
        manifest = instance.state_dir / record["input_evidence"]["manifest"]
        assert (
            json.loads(manifest.read_text())["files"]["submission.csv"]["sha256"]
            == hashlib.sha256(b"original").hexdigest()
        )
        assert (manifest.parent / "output/submission.csv").read_text() == "original"


def test_mutation_refuses_scoring_and_records_failure(monkeypatch, tmp_path):
    calls = []
    instance = service(monkeypatch, tmp_path, lambda path: calls.append(path))
    original = snapshots._copy_file

    def mutate(fd, target, deadline):
        result = original(fd, target, deadline)
        (instance.output_dir / "submission.csv").write_text("changed")
        return result

    monkeypatch.setattr(snapshots, "_copy_file", mutate)
    with pytest.raises(snapshots.SnapshotCaptureError, match="changed during copy"):
        instance.evaluate()
    assert calls == []
    record = json.loads(instance.submissions_path.read_text())
    assert record["failure_kind"] == "evaluation_input_snapshot_failed"
    assert record["raw_scores"] is None
    manifest = json.loads((instance.state_dir / record["input_evidence"]["manifest"]).read_text())
    assert manifest["status"] == "failed"


@pytest.mark.parametrize("kind", ["file", "directory", "root", "hardlink"])
def test_links_refused(tmp_path, kind):
    source = (tmp_path / "source").resolve()
    source.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("not-public")
    if kind == "root":
        source.rmdir()
        source.symlink_to(outside, target_is_directory=True)
    elif kind == "hardlink":
        (source / "alias").hardlink_to(outside / "secret")
    else:
        (source / "alias").symlink_to(outside if kind == "directory" else outside / "secret")
    with pytest.raises(snapshots.SnapshotCaptureError):
        snapshots.capture_inputs(source, tmp_path / "snapshot", deadline=time.monotonic() + 10)
    assert not list((tmp_path / "snapshot/output").rglob("secret"))


@pytest.mark.parametrize("error", [RuntimeError("evaluator failed"), TimeoutError("3600 expired")])
def test_error_retains_manifest(monkeypatch, tmp_path, error):
    def evaluate(path):
        raise error

    instance = service(monkeypatch, tmp_path, evaluate)
    with pytest.raises(type(error), match=str(error)):
        instance.evaluate()
    record = json.loads(instance.submissions_path.read_text())
    manifest = instance.state_dir / record["input_evidence"]["manifest"]
    assert json.loads(manifest.read_text())["status"] == "complete"
    assert (manifest.parent / "output/submission.csv").read_text() == "original"
    assert record["failure_kind"] == "evaluator_error"


def test_packaging_adds_export_and_preserves_science(tmp_path):
    import tomllib

    task = tmp_path / MOLECULAR_TASK
    module = task / "environment/sidecar/naturebench_eval_service/evaluation.py"
    module.parent.mkdir(parents=True)
    module.write_bytes(FIXTURE.read_bytes())
    (module.parent.parent / "Dockerfile").write_text(
        "COPY naturebench_eval_service /opt/naturebench/naturebench_eval_service\n"
    )
    config = task / "task.toml"
    config.write_text(
        'artifacts = [\n  { source = "/state/submissions.jsonl", service = "naturebench-eval" },\n]\n[agent]\ntimeout_sec=43200\n'
    )
    with config.open("a") as stream:
        stream.write(
            '[[verifier.collect]]\nservice="naturebench-eval"\ncommand="/opt/naturebench/bin/drain-and-finalize"\ntimeout_sec=3700\n'
        )
    before = tomllib.loads(config.read_text())
    with pytest.raises(ValueError, match="positive overhead"):
        install_snapshot_variant(task, capture_timeout_seconds=100)
    assert module.read_bytes() == FIXTURE.read_bytes()
    install_snapshot_variant(task, capture_timeout_seconds=60)
    after = tomllib.loads(config.read_text())
    assert after["artifacts"].pop() == {"source": "/state/evaluation_inputs", "service": "naturebench-eval"}
    assert after == before
    assert "deadline = time.monotonic() + 60.0" in module.read_text()
    assert "timeout=3600" in module.read_text()
    assert module.with_name("evaluation_inputs.py").read_bytes() == Path(snapshots.__file__).read_bytes()
    with pytest.raises(ValueError, match="Unqualified"):
        patch_evaluation(module.read_text(), capture_timeout_seconds=60)


def test_directory_replacement_race_refused(monkeypatch, tmp_path):
    source = (tmp_path / "source").resolve()
    (source / "nested").mkdir(parents=True)
    (source / "nested/public").write_text("original")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("private")
    original = snapshots._copy_file

    def replace(fd, target, deadline):
        result = original(fd, target, deadline)
        (source / "nested").rename(tmp_path / "old")
        (source / "nested").symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(snapshots, "_copy_file", replace)
    with pytest.raises(snapshots.SnapshotCaptureError, match="link"):
        snapshots.capture_inputs(source, tmp_path / "snapshot", deadline=time.monotonic() + 10)
    assert not list((tmp_path / "snapshot/output").rglob("secret"))
    assert json.loads((tmp_path / "snapshot/manifest.json").read_text())["status"] == "failed"


def test_deadline_during_copy_never_scores(monkeypatch, tmp_path):
    calls = []
    instance = service(monkeypatch, tmp_path, lambda path: calls.append(path))
    clock = [0.0]
    monkeypatch.setattr(snapshots.time, "monotonic", lambda: clock[0])
    original = snapshots._copy_file

    def expire(fd, target, deadline):
        result = original(fd, target, deadline)
        clock[0] = 3601.0
        return result

    monkeypatch.setattr(snapshots, "_copy_file", expire)
    with pytest.raises(snapshots.SnapshotCaptureError, match="deadline exhausted"):
        instance.evaluate()
    assert calls == []
    record = json.loads(instance.submissions_path.read_text())
    assert record["failure_kind"] == "evaluation_input_snapshot_failed"
    assert json.loads((instance.state_dir / record["input_evidence"]["manifest"]).read_text())["status"] == "failed"


def test_insufficient_disk_never_scores(monkeypatch, tmp_path):
    calls = []
    instance = service(monkeypatch, tmp_path, lambda path: calls.append(path))
    monkeypatch.setattr(snapshots.shutil, "disk_usage", lambda _: types.SimpleNamespace(free=0))
    with pytest.raises(snapshots.SnapshotCaptureError, match="disk space"):
        instance.evaluate()
    assert calls == []
    assert json.loads(instance.submissions_path.read_text())["failure_kind"] == "evaluation_input_snapshot_failed"


def test_manifest_atomic_publication_failure_never_scores(monkeypatch, tmp_path):
    calls = []
    instance = service(monkeypatch, tmp_path, lambda path: calls.append(path))

    def fail_replace(*args):
        raise OSError("publication fixture failure")

    monkeypatch.setattr(snapshots.os, "replace", fail_replace)
    with pytest.raises(snapshots.SnapshotCaptureError, match="publication failed"):
        instance.evaluate()
    assert calls == []
    assert json.loads(instance.submissions_path.read_text())["failure_kind"] == "evaluation_input_snapshot_failed"
    assert not (instance.state_dir / "evaluation_inputs/1/manifest.json").exists()
    assert (instance.state_dir / "evaluation_inputs/1/.manifest.tmp").is_file()


def test_slow_capture_preserves_full_scientific_timeout(monkeypatch, tmp_path):
    instance = service(monkeypatch, tmp_path, lambda _: {})
    module_globals = instance.evaluate.__func__.__globals__
    clock = [0.0]
    monkeypatch.setattr(snapshots.time, "monotonic", lambda: clock[0])
    original = snapshots._copy_file

    def slow_copy(fd, target, deadline):
        result = original(fd, target, deadline)
        clock[0] = 30.0
        return result

    monkeypatch.setattr(snapshots, "_copy_file", slow_copy)
    timeouts = []

    def subprocess_run(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        return types.SimpleNamespace(returncode=0, stdout='===EVAL_RESULT_JSON==={"fixture": true}', stderr="")

    monkeypatch.setattr(module_globals["subprocess"], "run", subprocess_run)
    instance._evaluator = lambda output: module_globals["run_evaluator_subprocess"](tmp_path, output)
    instance.evaluate()
    assert timeouts == [3600]


@pytest.mark.parametrize("allowance", [0, -1, float("inf"), float("nan"), True])
def test_capture_allowance_invalid(allowance):
    with pytest.raises(ValueError, match="finite and positive"):
        patch_evaluation(FIXTURE.read_text(), capture_timeout_seconds=allowance)


def test_prior_input_state_refused(monkeypatch, tmp_path):
    instance = service(monkeypatch, tmp_path, lambda _: {})
    instance.evaluate()
    manifest = instance.state_dir / "evaluation_inputs/1/manifest.json"
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match="Refuse reuse"):
        type(instance)(
            "fixture", tmp_path / "metadata", instance.output_dir, instance.state_dir, evaluator=lambda _: {}
        )
    assert manifest.read_bytes() == before


@pytest.mark.parametrize("enabled", [False, True])
def test_materialization_opt_in_and_original_default(monkeypatch, tmp_path, enabled):
    import tarfile
    import tomllib

    from benchmarks.naturebench import prepare_runtime

    task = tmp_path / "original" / MOLECULAR_TASK
    module = task / "environment/sidecar/naturebench_eval_service/evaluation.py"
    module.parent.mkdir(parents=True)
    module.write_bytes(FIXTURE.read_bytes())
    (module.parent.parent / "Dockerfile").write_text(
        "COPY naturebench_eval_service /opt/naturebench/naturebench_eval_service\n"
    )
    (task / "tests").mkdir()
    (task / "tests/judge_readme.py").write_text("fixture judge")
    scorer = task / "environment/sidecar/evaluation/evaluator.py"
    scorer.parent.mkdir()
    scorer.write_text("# scientific scorer remains untouched\n")
    (task / "task.toml").write_text(
        'artifacts = [\n  { source = "/state/submissions.jsonl", service = "naturebench-eval" },\n]\n[[verifier.collect]]\nservice="naturebench-eval"\ncommand="/opt/naturebench/bin/drain-and-finalize"\ntimeout_sec=3700\n'
    )
    archive = tmp_path / "task.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(task, arcname=task.name)
    monkeypatch.setattr(prepare_runtime, "patch_judge", lambda source: source)
    target = tmp_path / "prepared" / task.name
    prepare_runtime.materialize_archive(
        archive,
        target,
        expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        snapshot_evaluation_inputs=enabled,
        snapshot_capture_timeout_seconds=60 if enabled else None,
    )
    receipt = json.loads(target.with_suffix(".source.json").read_text())
    assert receipt["evaluation_input_snapshot_variant"] is enabled
    assert receipt["snapshot_capture_timeout_seconds"] == (60 if enabled else None)
    assert (target / scorer.relative_to(task)).read_bytes() == scorer.read_bytes()
    assert (target / module.relative_to(task)).read_bytes() == (
        patch_evaluation(FIXTURE.read_text(), capture_timeout_seconds=60).encode() if enabled else FIXTURE.read_bytes()
    )
    artifacts = tomllib.loads((target / "task.toml").read_text())["artifacts"]
    assert any(a["source"] == "/state/evaluation_inputs" for a in artifacts) is enabled
    assert tomllib.loads((target / "task.toml").read_text())["verifier"]["collect"][0]["timeout_sec"] == 3700
