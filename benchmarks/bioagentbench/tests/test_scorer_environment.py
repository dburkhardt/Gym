# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest

from benchmarks.bioagentbench import scorer_environment as scorer


@pytest.fixture
def qualified(tmp_path, monkeypatch):
    root = tmp_path / "environment"
    files = []
    for name in scorer.EXECUTABLES:
        path = root / "root/envs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = root / "transfer-manifest.json"
    manifest.write_text(json.dumps({"files": files}))
    monkeypatch.setattr(scorer, "ROOT", root)
    monkeypatch.setattr(scorer, "MANIFEST_SHA256", scorer.digest(manifest))
    return {"task_id": "giab", "giab_scorer_binding": scorer.binding("giab", manifest)}


def test_exact_binding_builds_clean_environment(qualified, monkeypatch):
    monkeypatch.setenv("PATH", "/untrusted")
    monkeypatch.setenv("PROVIDER_SECRET", "not-inherited")
    env = scorer.child_environment(qualified, home="/home/task", source="/src")
    assert env == {
        "PATH": str(scorer.ROOT / "root/envs/outer/bin") + ":/usr/bin:/bin",
        "MAMBA_ROOT_PREFIX": str(scorer.ROOT / "root"),
        "HOME": "/home/task",
        "PYTHONPATH": "/src",
    }


def test_ordinary_task_unchanged():
    assert scorer.child_environment({"task_id": "cystic-fibrosis"}, home="/h", source="/s") == {
        "PATH": "/usr/bin:/bin",
        "HOME": "/h",
        "PYTHONPATH": "/s",
    }
    with pytest.raises(ValueError):
        scorer.binding("other", "/any/receipt")


@pytest.mark.parametrize("change", ["bytes", "executable", "extra", "manifest", "root", "task"])
def test_drift_refused(qualified, change):
    if change == "bytes":
        (scorer.ROOT / "root/envs/hap/bin/samtools").write_bytes(b"changed")
    elif change == "executable":
        (scorer.ROOT / "root/envs/hap/bin/samtools").chmod(0o644)
    elif change == "extra":
        (scorer.ROOT / "root/envs/extra").write_text("extra")
    elif change == "manifest":
        (scorer.ROOT / "transfer-manifest.json").write_text("{}")
    elif change == "root":
        qualified["giab_scorer_binding"]["root"] = "/wrong"
    else:
        qualified["task_id"] = "other"
    with pytest.raises(ValueError):
        scorer.child_environment(qualified, home="/h", source="/s")


def test_missing_giab_binding_refused():
    with pytest.raises(ValueError):
        scorer.child_environment({"task_id": "giab"}, home="/h", source="/s")


def test_actual_child_environment_must_match(qualified, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv("MAMBA_ROOT_PREFIX", raising=False)
    with pytest.raises(ValueError, match="differs"):
        scorer.observed_environment(qualified)
    env = scorer.child_environment(qualified, home="/h", source="/s")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert scorer.observed_environment(qualified)["MAMBA_ROOT_PREFIX"] == env["MAMBA_ROOT_PREFIX"]
