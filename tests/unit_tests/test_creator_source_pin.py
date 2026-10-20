# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A matching HEAD is insufficient when a creator scorer has local modifications."""

import subprocess
from pathlib import Path

import pytest

from benchmarks.scientific_agent_campaign import deterministic


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def creator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    git(tmp_path, "init", "--quiet")
    (tmp_path / "scorer.py").write_text("def score(): return 1\n")
    git(tmp_path, "add", "scorer.py")
    git(tmp_path, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "source")
    monkeypatch.setitem(deterministic.PINS, "test_creator", git(tmp_path, "rev-parse", "HEAD"))
    return tmp_path


def test_clean_creator_and_untracked_receipt(creator: Path) -> None:
    deterministic.validate_repo("test_creator", creator)
    (creator / "receipt.json").write_text("{}")
    deterministic.validate_repo("test_creator", creator)


@pytest.mark.parametrize("change", ["unstaged", "staged", "deleted"])
def test_modified_creator_rejected_despite_matching_head(creator: Path, change: str) -> None:
    source = creator / "scorer.py"
    if change == "deleted":
        source.unlink()
    else:
        source.write_text("def score(): return 0\n")
    if change == "staged":
        git(creator, "add", "scorer.py")
    with pytest.raises(ValueError, match="differs from its pinned revision"):
        deterministic.validate_repo("test_creator", creator)
