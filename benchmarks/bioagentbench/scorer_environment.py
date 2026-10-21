# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GIAB-only admission of the preserved native hap environment; never installs."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath


ROOT = Path("/home/ubuntu/projects/broad-eval-campaign/bioagent-giab-environment")
MANIFEST_SHA256 = "00db69c4c57a13f6a0089d2b1f1d61e30c9da39e828e9e46d34b989e165e4d43"
EXECUTABLES = ("outer/bin/mamba", "hap/bin/python", "hap/bin/hap.py", "hap/bin/samtools", "hap/bin/bcftools")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binding(task_id, manifest):
    if task_id != "giab":
        if manifest is not None:
            raise ValueError("GIAB scorer binding is not valid for another task")
        return None
    if manifest is None or Path(manifest) != ROOT / "transfer-manifest.json":
        raise ValueError("GIAB requires the qualified absolute scorer manifest")
    return {"manifest": str(manifest), "sha256": MANIFEST_SHA256, "root": str(ROOT)}


def validate(packet):
    admitted = packet.get("giab_scorer_binding")
    expected = binding(packet["task_id"], admitted.get("manifest") if isinstance(admitted, dict) else None)
    if admitted != expected:
        raise ValueError("GIAB scorer binding changed")
    if expected is None:
        return None
    manifest = Path(expected["manifest"])
    if any(p.is_symlink() for p in (ROOT, *ROOT.parents, manifest)) or not manifest.is_file():
        raise ValueError("Scorer root/manifest must be regular owned paths")
    if digest(manifest) != MANIFEST_SHA256:
        raise ValueError("GIAB scorer inventory changed")
    entries = json.loads(manifest.read_text())["files"]
    names = set()
    for item in entries:
        name = item["path"]
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("root", "envs") or name in names:
            raise ValueError("Invalid scorer inventory path")
        names.add(name)
        path = ROOT / name
        if any(p.is_symlink() for p in path.parents):
            raise ValueError("Scorer file traverses symlink directory")
        if item.get("type") == "symlink":
            if not path.is_symlink() or os.readlink(path) != item["target"] or not path.resolve().is_relative_to(ROOT):
                raise ValueError("Scorer symlink changed or escapes root")
        elif (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item["bytes"]
            or digest(path) != item["sha256"]
        ):
            raise ValueError("Scorer file changed: " + name)
    actual = {str(p.relative_to(ROOT)) for p in (ROOT / "root/envs").rglob("*") if p.is_file() or p.is_symlink()}
    if names != actual:
        raise ValueError("Scorer inventory membership changed")
    for name in EXECUTABLES:
        executable = ROOT / "root/envs" / name
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError("Scorer executable unavailable: " + name)
    return expected


def child_environment(packet, *, home, source):
    admitted = validate(packet)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "PYTHONPATH": str(source)}
    if admitted is not None:
        env["PATH"] = str(ROOT / "root/envs/outer/bin") + ":" + env["PATH"]
        env["MAMBA_ROOT_PREFIX"] = str(ROOT / "root")
    return env


def observed_environment(packet):
    """Check the actual child lookup environment, not merely its intended packet."""
    expected = child_environment(packet, home=os.environ.get("HOME", ""), source=os.environ.get("PYTHONPATH", ""))
    for key in ("PATH", "MAMBA_ROOT_PREFIX"):
        if os.environ.get(key) != expected.get(key):
            raise ValueError("Creator scorer environment differs: " + key)
    return {key: expected[key] for key in ("PATH", "MAMBA_ROOT_PREFIX") if key in expected}
