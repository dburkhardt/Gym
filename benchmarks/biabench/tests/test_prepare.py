# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest

from benchmarks.biabench import prepare


@pytest.mark.parametrize("kind", ["reference", "symlink"])
def test_solver_contamination_prevents_publication(tmp_path, monkeypatch, kind):
    source, task, destination = (tmp_path / n for n in ("source", "task", "prepared"))
    source.mkdir()
    task.mkdir()
    (destination / "solver").mkdir(parents=True)
    reference = task / "evaluation_rubric.yaml"
    reference.write_text("private reference")
    leaked = destination / "solver" / reference.name
    if kind == "symlink":
        leaked.symlink_to(reference)
    else:
        leaked.write_text(reference.read_text())
    manifest = tmp_path / "frozen_sources.json"
    manifest.write_text(
        json.dumps(
            {"creator_files": {}, "task_files": {reference.name: hashlib.sha256(reference.read_bytes()).hexdigest()}}
        )
    )
    monkeypatch.setattr(prepare, "__file__", str(tmp_path / "prepare.py"))
    with pytest.raises(ValueError, match="solver|Solver"):
        prepare.stage(source, task, destination)
    assert not (destination / "benchmark.jsonl").exists()


def test_changed_source_rejected(tmp_path):
    asset = tmp_path / "file"
    asset.write_bytes(b"original")
    pinned = {"file": hashlib.sha256(asset.read_bytes()).hexdigest()}
    prepare.validate(tmp_path, pinned)
    asset.write_bytes(b"changed")
    with pytest.raises(ValueError, match="mismatch"):
        prepare.validate(tmp_path, pinned)


@pytest.mark.parametrize("role", ["solver", "references", "verifier_source"])
def test_role_directory_alias_rejected_before_staging(tmp_path, monkeypatch, role):
    source, task, destination = (tmp_path / n for n in ("source", "task", "prepared"))
    source.mkdir()
    task.mkdir()
    destination.mkdir()
    outside = tmp_path / "shared"
    outside.mkdir()
    (destination / role).symlink_to(outside, target_is_directory=True)
    (tmp_path / "frozen_sources.json").write_text(json.dumps({"creator_files": {}, "task_files": {}}))
    monkeypatch.setattr(prepare, "__file__", str(tmp_path / "prepare.py"))
    with pytest.raises(ValueError, match="role must not be a symlink"):
        prepare.stage(source, task, destination)
    assert list(outside.iterdir()) == []
    assert not (destination / "benchmark.jsonl").exists()
