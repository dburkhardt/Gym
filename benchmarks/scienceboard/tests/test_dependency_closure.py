# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real preserved creator inputs: dependency/image-overlay tampering fails before publication."""

import json
import shutil

import pytest

from benchmarks.scienceboard.prepare import MANIFEST, prepare
from benchmarks.scienceboard.tests.test_managed import SOURCE


DEPENDENCIES = [
    "requirements.txt",
    ".dockerignore",
    "docker/Dockerfile",
    "docker/import_rootfs.sh",
    "docker/flatten_image.sh",
    "docker/export/export_reference.py",
    "scripts/audit_distribution.py",
    "scripts/package_distribution.sh",
    "vm_config/server.py",
    "vm_config/pyxcursor.py",
    "docker/reset.sh",
    "docker/entrypoint.sh",
    "docker/session.sh",
    "docker/code-wrapper.sh",
    "docker/firefox-wrapper.sh",
    "docker/firefox.desktop",
    "docker/firefox-policies.json",
]


@pytest.fixture
def creator_copy(tmp_path):
    if not SOURCE.is_dir():
        pytest.skip("Pinned creator source absent")
    source = tmp_path / "creator"
    manifest = json.loads(MANIFEST.read_text())
    names = set(manifest["files"]) | {task["path"] for task in manifest["tasks"]}
    for relative in names:
        dest = source / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / relative, dest)
    return source


@pytest.mark.parametrize("relative", DEPENDENCIES)
def test_real_dependency_or_overlay_mutation_refused_before_publication(creator_copy, tmp_path, relative):
    original = creator_copy / relative
    original.write_bytes(original.read_bytes() + b"\n# altered runtime input\n")
    destination = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Creator source changed"):
        prepare(creator_copy, destination)
    assert not destination.exists()


@pytest.mark.parametrize("relative", ["requirements.txt", "docker/entrypoint.sh", "vm_config/server.py"])
def test_runtime_factory_rechecks_dependency_closure(creator_copy, monkeypatch, relative):
    from benchmarks.scienceboard.managed_desktop import creator_desktop_factory

    monkeypatch.setattr(
        "benchmarks.scienceboard.admission.admit", lambda *args: {"blockers": [], "image_id": "fixture"}
    )
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        monkeypatch.delenv(key, raising=False)
    (creator_copy / relative).write_bytes(b"altered")
    with pytest.raises(ValueError, match="Creator source changed"):
        creator_desktop_factory(creator_copy, metadata={}, audit={}, resources={})
