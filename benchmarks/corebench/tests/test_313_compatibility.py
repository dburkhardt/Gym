# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.corebench import image_controls as controls
from benchmarks.corebench.prepare import RUNTIME_SHA256
from benchmarks.corebench.prepare_frozen20 import validate_qualification
from benchmarks.corebench.qualification_compatibility import PRE_313_ENGINE, reviewed_313_addition


def bound():
    row = next(
        r
        for r in json.loads((Path(controls.__file__).with_name("frozen20.json")).read_text())["rows"]
        if r["id"] == "capsule-6746514"
    )
    payload = {
        "implementation_sha256": PRE_313_ENGINE,
        "schema": "corebench-image-qualification-v2",
        "recipe": controls.recipe_digest(row),
        "source": controls.source_digest(row),
        "instruction": row["instruction_sha256"],
        "reference": row["reference_sha256"],
        "scorer": row["scorer_sha256"],
        "harness_archive": RUNTIME_SHA256,
        "scientific_probe": controls.SCIENTIFIC_PROBE,
        "harness_probe": controls.HARNESS_PROBE,
    }
    receipt = {
        "capsule": row["id"],
        "creator_image": row["published_image"],
        "recipe_sha256": controls.recipe_digest(row),
        "source_sha256": controls.source_digest(row),
        "qualification_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "final_image": "sha256:" + "a" * 64,
        "scientific_before": "actual",
        "scientific_after": "actual",
        "image_source_modes": {x["path"]: 420 for x in row["medium_file_manifest"]},
        "resources": {"cpus": 2, "memory_mb": 8192, "storage_mb": 10000, "agent_timeout_sec": 3600},
    }
    for k in [
        "harness_python_ok",
        "tmux_ok",
        "asciinema_ok",
        "reference_isolation_ok",
        "cleanup_ok",
        "image_source_verified",
    ]:
        receipt[k] = True
    return row, receipt


def test_ordinary_validation_uses_exact_transition_without_mutation():
    row, receipt = bound()
    original = copy.deepcopy(receipt)
    assert reviewed_313_addition(row, receipt)
    validate_qualification(row, receipt)
    assert receipt == original


@pytest.mark.parametrize("key", ["instruction_sha256", "reference_sha256", "scorer_sha256", "published_image", "id"])
def test_changed_task_or_other_capsule_rejected(key):
    row, receipt = bound()
    row[key] = "changed"
    with pytest.raises(ValueError):
        validate_qualification(row, receipt)


@pytest.mark.parametrize(
    "key,value",
    [
        ("cleanup_ok", False),
        ("image_source_verified", False),
        ("scientific_after", "changed"),
        ("qualification_sha256", "0" * 64),
        ("recipe_sha256", "0" * 64),
        ("image_source_modes", {}),
    ],
)
def test_no_ordinary_receipt_check_bypassed(key, value):
    row, receipt = bound()
    receipt[key] = value
    with pytest.raises(ValueError):
        validate_qualification(row, receipt)


def test_changed_source_probe_or_engine_rejected(monkeypatch, tmp_path):
    row, receipt = bound()
    row["medium_file_manifest"][0]["sha256"] = "a" * 64
    with pytest.raises(ValueError):
        validate_qualification(row, receipt)
    row, receipt = bound()
    monkeypatch.setattr(controls, "SCIENTIFIC_PROBE", "changed")
    assert reviewed_313_addition(row, receipt) is None
    path = tmp_path / "changed.py"
    path.write_text("changed")
    monkeypatch.setattr(controls, "__file__", str(path))
    assert reviewed_313_addition(row, receipt) is None
