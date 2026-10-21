# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest

from benchmarks.corebench.image_controls import dockerfile, qualification_digest, recipe_digest, source_digest
from benchmarks.corebench.prepare_frozen20 import stage, validate_qualification


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
        "qualification_sha256": qualification_digest(row),
        "scientific_before": "python3 creator",
        "scientific_after": "python3 creator",
        "harness_python_ok": True,
        "tmux_ok": True,
        "asciinema_ok": True,
        "reference_isolation_ok": True,
        "cleanup_ok": True,
        "image_source_verified": True,
        "image_source_modes": {"code/run.py": 0o644},
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


@pytest.mark.parametrize("field", ["instruction_sha256", "reference_sha256", "scorer_sha256"])
def test_qualification_binds_task_and_reference(tmp_path, field):
    _, _, row, receipt, _ = fixture(tmp_path)
    row[field] = "f" * 64
    with pytest.raises(ValueError, match="binding missing or stale"):
        validate_qualification(row, receipt)


def test_qualification_binds_probe_and_refuses_historical_upgrade(tmp_path, monkeypatch):
    import benchmarks.corebench.image_controls as controls

    _, _, row, receipt, _ = fixture(tmp_path)
    historical = dict(receipt)
    historical.pop("qualification_sha256")
    with pytest.raises(ValueError, match="binding missing or stale"):
        validate_qualification(row, historical)
    monkeypatch.setattr(controls, "SCIENTIFIC_PROBE", controls.SCIENTIFIC_PROBE + "\necho changed\n")
    with pytest.raises(ValueError, match="binding missing or stale"):
        validate_qualification(row, receipt)


def test_r_probe_records_packages_and_library_paths():
    from benchmarks.corebench.image_controls import SCIENTIFIC_PROBE

    assert "installed.packages()" in SCIENTIFIC_PROBE
    assert ".libPaths()" in SCIENTIFIC_PROBE
    assert '"Package", "Version", "LibPath", "Built"' in SCIENTIFIC_PROBE


def test_qualification_binds_recovered_runtime(tmp_path, monkeypatch):
    import benchmarks.corebench.prepare as preparation

    _, _, row, receipt, _ = fixture(tmp_path)
    monkeypatch.setattr(preparation, "RUNTIME_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="binding missing or stale"):
        validate_qualification(row, receipt)


@pytest.mark.asyncio
async def test_maintained_qualification_producer(tmp_path, monkeypatch):
    import benchmarks.corebench.image_controls as controls
    import benchmarks.corebench.prepare as preparation

    medium, refs, row, _, _ = fixture(tmp_path)
    runtime = tmp_path / "runtime.tar"
    runtime.write_bytes(b"fixture runtime")
    monkeypatch.setattr(preparation, "RUNTIME_SHA256", hashlib.sha256(runtime.read_bytes()).hexdigest())
    calls = []

    async def probe(image, command, destination, **kwargs):
        calls.append((image, command, kwargs))
        return {
            "image": {"Id": "sha256:" + "b" * 64},
            "stdout": "harness-python-ok reference-isolation-ok"
            if command == controls.HARNESS_PROBE
            else ("image-source-ok" if "image-source-ok" in command else "R package inventory"),
        }

    monkeypatch.setattr(controls, "probe_image", probe)
    receipt = await controls.qualify_images(
        row,
        medium,
        refs,
        runtime,
        "sha256:" + "b" * 64,
        tmp_path / "qualified",
        resources={"cpus": 2, "memory_mb": 4096, "storage_mb": 10240, "agent_timeout_sec": 3600},
    )
    validate_qualification(row, receipt)
    assert len(calls) == 4
    assert calls[0][1] == calls[1][1] == controls.SCIENTIFIC_PROBE
    assert json.loads((tmp_path / "qualified/qualification.json").read_text()) == receipt


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["source", "identity"])
async def test_producer_refuses_wrong_image_source_or_identity(tmp_path, monkeypatch, failure):
    import benchmarks.corebench.image_controls as controls
    import benchmarks.corebench.prepare as preparation

    medium, refs, row, _, _ = fixture(tmp_path)
    runtime = tmp_path / "runtime.tar"
    runtime.write_bytes(b"fixture runtime")
    monkeypatch.setattr(preparation, "RUNTIME_SHA256", hashlib.sha256(runtime.read_bytes()).hexdigest())

    async def probe(image, command, destination, **kwargs):
        if destination.name == "image-source" and failure == "source":
            raise ValueError("image source inventory differs")
        return {
            "image": {
                "Id": "sha256:" + ("c" if destination.name == "harness" and failure == "identity" else "b") * 64
            },
            "stdout": "harness-python-ok reference-isolation-ok"
            if command == controls.HARNESS_PROBE
            else ("image-source-ok" if destination.name == "image-source" else "R inventory"),
        }

    monkeypatch.setattr(controls, "probe_image", probe)
    with pytest.raises(ValueError, match="inventory differs|identity changed"):
        await controls.qualify_images(
            row,
            medium,
            refs,
            runtime,
            "sha256:" + "b" * 64,
            tmp_path / "out",
            resources={"cpus": 2, "memory_mb": 4096, "storage_mb": 10240, "agent_timeout_sec": 3600},
        )
    assert not (tmp_path / "out/qualification.json").exists()


@pytest.mark.parametrize("mutation", [None, "bytes", "mode", "extra"])
def test_actual_image_source_check(tmp_path, mutation):
    import subprocess
    import sys

    from benchmarks.corebench.image_controls import image_source_probe

    medium, _, row, _, _ = fixture(tmp_path)
    root = medium / row["id"]
    file = root / "code/run.py"
    modes = {"code/run.py": file.stat().st_mode & 0o777}
    command = image_source_probe(row, modes, root)
    if mutation == "bytes":
        file.write_text("changed")
    elif mutation == "mode":
        file.chmod(0o700)
    elif mutation == "extra":
        (root / "unexpected").write_text("extra")
    result = subprocess.run([sys.executable, "-O", "-c", command], capture_output=True, text=True)
    assert (result.returncode == 0) is (mutation is None)
    assert ("image-source-ok" in result.stdout) is (mutation is None)
