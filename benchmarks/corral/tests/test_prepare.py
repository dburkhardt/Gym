# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest

from benchmarks.corral import prepare


def fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    relative = "tasks/retrosynthesis/environments/level_3/tasks_json/make_1.json"
    path = source / relative
    path.parent.mkdir(parents=True)
    native = {"id": "make_1_lvl3", "input": {"prompt": "Exact native prompt"}, "output": [{"target": 17}]}
    path.write_text(json.dumps([native]))
    manifest = {
        "source_commit": "pinned",
        "frozen5": ["make_1_lvl3"],
        "files": {relative: hashlib.sha256(path.read_bytes()).hexdigest()},
    }
    (tmp_path / "frozen_sources.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(prepare, "__file__", str(tmp_path / "prepare.py"))
    return source, path, native


def test_exact_native_fields_and_provenance(tmp_path, monkeypatch):
    source, path, native = fixture(tmp_path, monkeypatch)
    result = prepare.stage(source, tmp_path / "out")
    row = json.loads(result.read_text())
    assert row["verifier_metadata"]["native_task"] == native
    assert row["task_id"] == native["id"]
    assert row["verifier_metadata"]["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert row["responses_create_params"]["input"] == []


def test_source_mismatch_fails_before_publish(tmp_path, monkeypatch):
    source, path, _ = fixture(tmp_path, monkeypatch)
    path.write_text("[]")
    with pytest.raises(ValueError, match="source mismatch"):
        prepare.stage(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
