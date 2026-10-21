# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.corebench import image_controls as controls
from benchmarks.corebench.qualification_compatibility import PRIOR_QUALIFICATION, reviewed_compatibility


def inputs():
    row = next(
        r
        for r in json.loads((Path(controls.__file__).parent / "frozen20.json").read_text())["rows"]
        if r["id"] == "capsule-8197429"
    )
    return row, {"qualification_sha256": PRIOR_QUALIFICATION}


def test_exact_review_preserves_original_receipt():
    row, receipt = inputs()
    original = copy.deepcopy(receipt)
    result = reviewed_compatibility(row, receipt)
    assert result["original_qualification_sha256"] == PRIOR_QUALIFICATION
    assert result["reviewed_implementation_sha256"] == hashlib.sha256(Path(controls.__file__).read_bytes()).hexdigest()
    assert receipt == original


@pytest.mark.parametrize("field", ["reference_sha256", "instruction_sha256", "scorer_sha256", "published_image"])
def test_changed_bound_field_refused(field):
    row, receipt = inputs()
    row[field] += "changed"
    assert reviewed_compatibility(row, receipt) is None


def test_changed_source_refused():
    row, receipt = inputs()
    row["medium_file_manifest"][0]["sha256"] = "0" * 64
    assert reviewed_compatibility(row, receipt) is None


@pytest.mark.parametrize("field", ["SCIENTIFIC_PROBE", "HARNESS_PROBE"])
def test_changed_probe_refused(monkeypatch, field):
    row, receipt = inputs()
    monkeypatch.setattr(controls, field, getattr(controls, field) + "changed")
    assert reviewed_compatibility(row, receipt) is None


def test_changed_recipe_refused(monkeypatch):
    row, receipt = inputs()
    monkeypatch.setattr(controls, "recipe_digest", lambda row: "0" * 64)
    assert reviewed_compatibility(row, receipt) is None


@pytest.mark.parametrize("change", ["capsule", "variant", "receipt", "implementation"])
def test_outside_review_refused(change, monkeypatch, tmp_path):
    row, receipt = inputs()
    if change == "capsule":
        row["id"] = "capsule-1906954"
    if change == "variant":
        row["harness_install_variant"] = "ubuntu-bionic-isolated-v1"
    if change == "receipt":
        receipt["qualification_sha256"] = "0" * 64
    if change == "implementation":
        p = tmp_path / "changed.py"
        p.write_text("changed")
        monkeypatch.setattr(controls, "__file__", str(p))
    assert reviewed_compatibility(row, receipt) is None


def test_changed_runtime_archive_refused(monkeypatch):
    from benchmarks.corebench import prepare

    row, receipt = inputs()
    monkeypatch.setattr(prepare, "RUNTIME_SHA256", "0" * 64)
    assert reviewed_compatibility(row, receipt) is None


def test_compatibility_does_not_skip_other_control_checks():
    from benchmarks.corebench.prepare_frozen20 import validate_qualification

    row, receipt = inputs()
    receipt.update(
        capsule=row["id"],
        creator_image=row["published_image"],
        recipe_sha256=controls.recipe_digest(row),
        source_sha256=controls.source_digest(row),
        final_image="sha256:" + "a" * 64,
        scientific_before="same",
        scientific_after="same",
        image_source_modes={f["path"]: 0o755 for f in row["medium_file_manifest"]},
        resources={"cpus": 2, "memory_mb": 4096, "storage_mb": 10240, "agent_timeout_sec": 3600},
    )
    flags = [
        "harness_python_ok",
        "tmux_ok",
        "asciinema_ok",
        "reference_isolation_ok",
        "cleanup_ok",
        "image_source_verified",
    ]
    receipt.update({f: True for f in flags})
    original = copy.deepcopy(receipt)
    validate_qualification(row, receipt)
    assert receipt == original
    for flag in flags:
        changed = copy.deepcopy(receipt)
        changed[flag] = False
        with pytest.raises(ValueError, match="control incomplete"):
            validate_qualification(row, changed)
