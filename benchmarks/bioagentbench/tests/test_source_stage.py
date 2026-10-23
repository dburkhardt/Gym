# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess

import pytest

from benchmarks.bioagentbench import source_stage


def test_original_checkout_preflight_and_exported_snapshot_are_distinct(tmp_path, monkeypatch):
    source = tmp_path / "checkout"
    source.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args]).decode().strip()

    git("init", "-q")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Fixture")
    (source / "module.py").write_text("value = 1\n")
    git("add", "module.py")
    git("commit", "-qm", "fixture")
    revision = git("rev-parse", "HEAD")
    monkeypatch.setattr(source_stage, "CREATOR_REVISION", revision)
    assert source_stage.validate_creator_root(source)["revision"] == revision
    exported = source_stage.snapshot_creator(source, tmp_path / "export")
    assert (exported / "module.py").read_bytes() == (source / "module.py").read_bytes()
    with pytest.raises(ValueError, match="original pinned Git checkout"):
        source_stage.validate_creator_root(exported)
    with pytest.raises(ValueError, match="original pinned Git checkout"):
        source_stage.snapshot_creator(exported, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
    (source / "module.py").write_text("modified\n")
    with pytest.raises(ValueError, match="modifications"):
        source_stage.validate_creator_root(source)
