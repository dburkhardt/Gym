# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from types import SimpleNamespace

import pytest

from benchmarks.bioagentbench import worker


def test_owned_copy_preserves_bytes_and_original_reference(tmp_path, monkeypatch):
    source = tmp_path / "source" / "workflow"
    for category in ["data", "reference", "results"]:
        (source / category).mkdir(parents=True)
        (source / category / "fixture").write_bytes(category.encode())
    files = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": {"workflow": {"files": files}}}))
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3))
    packet = {
        "task_id": "workflow",
        "asset_manifest": str(manifest),
        "asset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    config = {"taskset": {"data_root": str(source.parent)}}
    selected = worker.stage_owned_inputs(packet, config, output)
    staged = output / "owned-inputs/workflow"
    assert selected["taskset"]["data_root"] == str(staged.parent)
    assert config["taskset"]["data_root"] == str(source.parent)
    assert {str(path.relative_to(staged)) for path in staged.rglob("*") if path.is_file()} == set(files)
    (staged / "reference/fixture").write_text("created scientific index")
    assert (source / "reference/fixture").read_text() == "reference"


def test_disk_gate_keeps_full_scope(tmp_path, monkeypatch):
    source = tmp_path / "source/workflow/data"
    source.mkdir(parents=True)
    (source / "fixture").write_text("data")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": {"workflow": {"files": {"data/fixture": "unused"}}}}))
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda _: SimpleNamespace(free=8 * 1024**3))
    with pytest.raises(RuntimeError, match="disk reserve"):
        worker.stage_owned_inputs(
            {
                "task_id": "workflow",
                "asset_manifest": str(manifest),
                "asset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            },
            {"taskset": {"data_root": str(source.parent.parent)}},
            tmp_path,
        )
    assert not (tmp_path / "owned-inputs").exists()
