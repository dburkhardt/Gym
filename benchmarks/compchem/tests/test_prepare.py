# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from benchmarks.compchem import prepare as module


@pytest.fixture
def release(tmp_path, monkeypatch):
    checkout, prepared = tmp_path / "source", tmp_path / "prepared"
    tasks = []
    for i in range(5):
        files = {
            "instruction.md": b"Solve the native task",
            "task.toml": b"version='1.0'",
            "environment/Dockerfile": b"FROM busybox",
            "tests/test.sh": b"native verifier",
            "tests/verify.py": b"assert scientific_result",
            "solution/solve.sh": b"oracle",
        }
        name = f"task-{i}"
        hashes = {}
        for relative, body in files.items():
            for base in (checkout, prepared):
                p = base / "tasks" / name / relative
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(body)
            hashes[relative] = module.digest(p)
        capture = prepared / "tasks" / name / "tests/capture_artifacts.py"
        capture.write_text("capture only")
        tasks.append(
            {
                "task": name,
                "source_files": hashes,
                "prepared_config_sha256": hashes["task.toml"],
                "prepared_dockerfile_sha256": hashes["environment/Dockerfile"],
                "prepared_test_sha256": hashes["tests/test.sh"],
                "artifact_capture_sha256": module.digest(capture),
            }
        )
    manifest = prepared / "manifest.json"
    manifest.write_text(json.dumps({"tasks": tasks}))
    monkeypatch.setattr(module, "MANIFEST_SHA256", module.digest(manifest))
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: module.SOURCE_COMMIT)
    return checkout, prepared, tmp_path / "result"


def test_resume_after_interrupted_publication_preserves_task_bytes(release, monkeypatch):
    checkout, prepared, dest = release
    real = module.stage_file

    def interrupt(source, target, **kwargs):
        if target.name == "benchmark.jsonl":
            raise OSError("interrupted before dataset publication")
        return real(source, target, **kwargs)

    monkeypatch.setattr(module, "stage_file", interrupt)
    with pytest.raises(OSError):
        module.stage(checkout, prepared, dest)
    assert not (dest / "benchmark.jsonl").exists()
    monkeypatch.setattr(module, "stage_file", real)
    path = module.stage(checkout, prepared, dest)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["instance_id"] for r in rows] == [f"compchem::task-{i}" for i in range(5)]
    assert (dest / "tasks/task-0/tests/verify.py").read_bytes() == (
        checkout / "tasks/task-0/tests/verify.py"
    ).read_bytes()


@pytest.mark.parametrize("change", ["creator", "overlay", "extra", "manifest"])
def test_tampering_fails_before_any_publication(release, change):
    checkout, prepared, dest = release
    if change == "creator":
        (checkout / "tasks/task-0/tests/verify.py").write_text("different scientific test")
    elif change == "overlay":
        (prepared / "tasks/task-0/tests/verify.py").write_text("different scientific test")
    elif change == "extra":
        (prepared / "tasks/task-0/unknown.txt").write_text("unrecorded")
    else:
        (prepared / "manifest.json").write_text("{}")
    with pytest.raises(ValueError):
        module.stage(checkout, prepared, dest)
    assert not dest.exists()


def test_scorer_controls_replace_plural_native_agents_without_model_route():
    from benchmarks.compchem.controls import control_config

    original = {
        "agents": [
            {"import_path": "Terminus2NemoGym", "model_name": "Nemotron", "kwargs": {"api_base": "live-provider"}}
        ],
        "datasets": [{"path": "native-tasks"}],
    }
    for name in ("nop", "oracle"):
        result = control_config(original, name)
        assert result["agents"] == [{"name": name}]
        assert "agent" not in result
        assert result["datasets"] == original["datasets"]
        assert "live-provider" not in json.dumps(result)
    assert original["agents"][0]["import_path"] == "Terminus2NemoGym"
    with pytest.raises(ValueError):
        control_config(original, "terminus-2")
