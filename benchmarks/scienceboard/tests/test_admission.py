# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import pytest

from benchmarks.scienceboard.admission import AUDIT_SHA256, SOURCE_REVISION, admit


def evidence():
    image = "sha256:" + "a" * 64
    metadata = {
        "Id": image,
        "Architecture": "amd64",
        "Os": "linux",
        "Config": {"Labels": {"org.scienceboard.distribution": "sanitized-flat"}, "Env": ["PATH=/usr/bin"]},
        "RootFS": {"Layers": ["sha256:" + "b" * 64]},
    }
    audit = {
        "source_revision": SOURCE_REVISION,
        "script_sha256": AUDIT_SHA256,
        "image_id": image,
        "returncode": 0,
        "report": {"image": image, "passed": True, "finding_count": 0, "findings": []},
    }
    return metadata, audit, {"cpus": 8, "memory_gib": 24, "free_disk_gib": 80, "shm_gib": 2}


def test_valid_evidence_is_not_runtime_acceptance():
    result = admit(*evidence())
    assert result["status"] == "image_evidence_admitted_runtime_unqualified"
    assert result["runnable"] is False
    assert result["remaining_gates"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("Id", "scienceboard:0.1"),
        ("Architecture", "arm64"),
        ("Os", "darwin"),
        ("RootFS", {"Layers": ["one", "two"]}),
        ("Config", {"Labels": {}}),
    ],
)
def test_wrong_image(field, value):
    m, a, r = evidence()
    m[field] = value
    assert admit(m, a, r)["status"] == "blocked"


def test_audit_identity_failure_and_missing_image():
    m, a, r = evidence()
    assert admit({}, a, r)["status"] == "blocked"
    a["report"]["image"] = "scienceboard:0.1"
    assert admit(m, a, r)["status"] == "blocked"


def test_audit_failure_resources_and_evidence_binding():
    m, a, r = evidence()
    original = admit(m, a, r)
    failed = copy.deepcopy(a)
    failed["returncode"] = 1
    assert admit(m, failed, r)["status"] == "blocked"
    r["memory_gib"] = 2
    changed = admit(m, a, r)
    assert changed["status"] == "blocked"
    assert changed["evidence_sha256"] != original["evidence_sha256"]
