# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from benchmarks.moosechem import isolated


@pytest.fixture
def public(tmp_path, monkeypatch):
    repo = tmp_path / "public"
    repo.mkdir()
    for name in isolated.PUBLIC_FILES:
        p = repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name)
    trusted = {name: isolated.digest(repo / name) for name in isolated.PUBLIC_FILES}
    monkeypatch.setattr(isolated, "pins", lambda: trusted)
    task = {"original_task_id": 26, "internal_task_id": 0, "question": "Exact question", "survey": "Exact survey"}
    (repo / "task.json").write_text(json.dumps(task))
    manifest = {name: isolated.digest(repo / name) for name in (*isolated.PUBLIC_FILES, "task.json")}
    (repo / "public-manifest.json").write_text(json.dumps(manifest))
    return repo, isolated.digest(repo / "public-manifest.json"), task


def test_exact_public_bundle_and_original_identity(public):
    repo, expected, task = public
    assert isolated.validate_public(repo, repo / "task.json", expected_manifest_sha256=expected) == task


def test_recomputed_task_manifest_does_not_change_trusted_binding(public):
    repo, expected, task = public
    task["original_task_id"] = 24
    (repo / "task.json").write_text(json.dumps(task))
    manifest = json.loads((repo / "public-manifest.json").read_text())
    manifest["task.json"] = isolated.digest(repo / "task.json")
    (repo / "public-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Trusted public manifest"):
        isolated.validate_public(repo, repo / "task.json", expected_manifest_sha256=expected)


def test_reference_workbook_cannot_be_added_to_public_tree(public):
    repo, expected, _ = public
    (repo / "Data/chem_research_2024.xlsx").write_text("reference fixture")
    with pytest.raises(ValueError, match="allowlist"):
        isolated.validate_public(repo, repo / "task.json", expected_manifest_sha256=expected)


def test_public_symlink_cannot_escape_mount(public, tmp_path):
    repo, expected, _ = public
    outside = tmp_path / "reference"
    outside.write_text("reference fixture")
    (repo / "escape").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        isolated.validate_public(repo, repo / "task.json", expected_manifest_sha256=expected)


def test_modified_creator_code_refused_even_with_rebound_manifest(public):
    repo, _, _ = public
    name = isolated.PUBLIC_FILES[0]
    (repo / name).write_text("modified creator")
    manifest = json.loads((repo / "public-manifest.json").read_text())
    manifest[name] = isolated.digest(repo / name)
    (repo / "public-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Public input changed"):
        isolated.validate_public(
            repo, repo / "task.json", expected_manifest_sha256=isolated.digest(repo / "public-manifest.json")
        )
