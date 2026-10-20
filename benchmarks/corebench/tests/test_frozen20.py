# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path
import pytest
from benchmarks.corebench.prepare_frozen20 import stage, validate_qualification
from benchmarks.corebench.image_controls import dockerfile, recipe_digest, source_digest


def fixture(tmp):
    medium = tmp / "medium"
    refs = tmp / "refs"
    cid = "capsule-test"
    (medium / cid / "code").mkdir(parents=True)
    (medium / cid / "code/run.py").write_text("print(1)")
    (refs / cid / "solver-input").mkdir(parents=True)
    (refs / cid / "evaluator-only").mkdir()
    files = {
        "solver-input/instruction.txt": "original task",
        "evaluator-only/answers.json": '[{"value":1}]',
        "evaluator-only/evaluations.py": "creator scorer",
    }
    for name, value in files.items():
        (refs / cid / name).write_text(value)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    row = {
        "id": cid,
        "published_image": "registry.test/source@sha256:" + "a" * 64,
        "runtime_environment_variant": {},
        "status": "assets_pinned_runtime_unmeasured",
        "medium_file_manifest": [{"path": "code/run.py", "sha256": digest(medium / cid / "code/run.py")}],
        "instruction_sha256": digest(refs / cid / "solver-input/instruction.txt"),
        "reference_sha256": digest(refs / cid / "evaluator-only/answers.json"),
        "scorer_sha256": digest(refs / cid / "evaluator-only/evaluations.py"),
    }
    q = {
        "capsule": cid,
        "creator_image": row["published_image"],
        "final_image": "sha256:" + "b" * 64,
        "recipe_sha256": recipe_digest(row),
        "source_sha256": source_digest(row),
        "scientific_before": "python3 creator",
        "scientific_after": "python3 creator",
        "harness_python_ok": True,
        "tmux_ok": True,
        "asciinema_ok": True,
        "reference_isolation_ok": True,
        "cleanup_ok": True,
        "resources": {"cpus": 2, "memory_mb": 4096, "storage_mb": 10240, "agent_timeout_sec": 3600},
    }
    manifest = tmp / "manifest.json"
    manifest.write_text(json.dumps({"rows": [row]}))
    return medium, refs, row, q, manifest


def test_sources_and_qualification_gate_publication(tmp_path):
    medium, refs, row, q, manifest = fixture(tmp_path)
    with pytest.raises(ValueError, match="No qualified"):
        stage(medium, refs, {}, tmp_path / "out", manifest=manifest)
    assert not (tmp_path / "out").exists()
    p = stage(medium, refs, {row["id"]: q}, tmp_path / "out", manifest=manifest)
    assert json.loads(p.read_text())["task_id"] == "corebench::capsule-test"
    task = p.parent / "tasks/capsule-test"
    assert not list((task / "environment").rglob("answers.json"))
    assert "/opt/corebench-harness/bin/python3" in (task / "tests/test.sh").read_text()
    assert json.loads((p.parent / "preparation.json").read_text())["retained"] == 1


def test_changed_science_or_source_rejected(tmp_path):
    medium, refs, row, q, manifest = fixture(tmp_path)
    q["scientific_after"] = "different python"
    with pytest.raises(ValueError, match="interpreter"):
        validate_qualification(row, q)
    q["scientific_after"] = q["scientific_before"]
    (medium / row["id"] / "code/run.py").write_text("changed")
    with pytest.raises(ValueError, match="Changed medium"):
        stage(medium, refs, {row["id"]: q}, tmp_path / "out", manifest=manifest)
    assert not (tmp_path / "out").exists()


def test_cuda_oom_and_recipe_require_explicit_controls(tmp_path):
    _, _, row, q, _ = fixture(tmp_path)
    row["status"] = "requires_cuda_no_cpu_substitution"
    with pytest.raises(ValueError, match="CUDA"):
        validate_qualification(row, q)
    row["status"] = "measured_oom_requires_resource_review"
    with pytest.raises(ValueError, match="OOM"):
        validate_qualification(row, q)
    q["oom_resource_review"] = True
    validate_qualification(row, q)
    q["recipe_sha256"] = "wrong"
    with pytest.raises(ValueError, match="recipe"):
        validate_qualification(row, q)


def test_recipe_never_replaces_scientific_python_path(tmp_path):
    _, _, row, _, _ = fixture(tmp_path)
    text = dockerfile(row)
    assert "ENV PATH" not in text
    assert "COPY python /opt/corebench-harness" in text
    assert "RUN python" not in text
    assert "/opt/corebench-harness/bin/python3 -m pip" in text


def test_unqualified_capsules_remain_visible(tmp_path):
    import shutil

    medium, refs, row, q, manifest = fixture(tmp_path)
    blocked = {**row, "id": "capsule-blocked", "status": "requires_cuda_no_cpu_substitution"}
    shutil.copytree(medium / row["id"], medium / blocked["id"])
    shutil.copytree(refs / row["id"], refs / blocked["id"])
    manifest.write_text(json.dumps({"rows": [row, blocked]}))
    p = stage(medium, refs, {row["id"]: q}, tmp_path / "out", manifest=manifest)
    receipt = json.loads((p.parent / "preparation.json").read_text())
    assert receipt["retained"] == 2 and receipt["published"] == 1
    assert receipt["blocked"] == [{"id": "capsule-blocked", "status": "requires_cuda_no_cpu_substitution"}]
