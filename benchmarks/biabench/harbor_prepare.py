# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize the one preserved CPU task without exposing verifier assets to Harbor."""

import hashlib
import json
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.biabench.prepare import stage, validate
from nemo_gym.dataset_staging import stage_file


SOLVER_IMAGE = "sha256:43e4fe86544cd2a980b0d7d9c10421d4458673a298203dc40aee30d324cfcf78"
TASK = "wound-healing-speed-kymograph-gigadb100118"


def solver_image() -> str:
    """Explicit qualified image identity; historical default remains unchanged."""
    image = os.environ.get("BIABENCH_SOLVER_IMAGE", SOLVER_IMAGE)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("BIABENCH_SOLVER_IMAGE requires an immutable full sha256 image ID")
    return image


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _published_names() -> list[str]:
    return [
        "benchmark.jsonl",
        f"tasks/{TASK}/instruction.md",
        f"tasks/{TASK}/task.toml",
        f"tasks/{TASK}/environment/Dockerfile",
        f"tasks/{TASK}/tests/test.sh",
    ]


def is_prepared_data_current(fpath: Path) -> bool:
    """Bind cache reuse to the requested image, pinned inputs and published bytes."""
    try:
        fpath = Path(fpath)
        receipt = json.loads((fpath.parent / "preparation.json").read_text())
        if receipt["image"] != solver_image() or receipt["prepare_sha256"] != _sha(Path(__file__)):
            return False
        manifest_path = Path(__file__).with_name("frozen_sources.json")
        manifest = json.loads(manifest_path.read_text())
        validate(Path(os.environ["BIABENCH_SOURCE"]), manifest["creator_files"])
        validate(Path(os.environ["BIABENCH_TASK"]), manifest["task_files"])
        if receipt["source_manifest_sha256"] != _sha(manifest_path):
            return False
        if set(receipt["files"]) != set(_published_names()):
            return False
        return all(
            not (fpath.parent / name).is_symlink() and _sha(fpath.parent / name) == digest
            for name, digest in receipt["files"].items()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare() -> Path:
    image = solver_image()  # Reject malformed overrides before staging any files.
    source, task, assets = (Path(os.environ[k]) for k in ("BIABENCH_SOURCE", "BIABENCH_TASK", "BIABENCH_DATA"))
    target = Path(__file__).parent / "data"
    if (target / "benchmark.jsonl").exists() and not is_prepared_data_current(target / "benchmark.jsonl"):
        raise ValueError("Stale BIABench prepared cache: use a fresh versioned source/preparation directory")
    stage(source, task, assets)
    root = target / "tasks" / TASK
    with TemporaryDirectory() as temp:
        temp = Path(temp)
        stage_file(assets / "solver/instruction.md", root / "instruction.md")
        definition = temp / "task.toml"
        definition.write_text(
            '''version = "1.0"
artifacts = ["/workspace"]
[metadata]
author_name = "BIABench authors; maintained native CPU variant"
[agent]
timeout_sec = 7200
[verifier]
environment_mode = "separate"
timeout_sec = 3720
[environment]
docker_image = "'''
            + image
            + """"
cpus = 4
memory_mb = 8192
storage_mb = 10240
allow_internet = false
"""
        )
        stage_file(definition, root / "task.toml")
        dockerfile = temp / "Dockerfile"
        dockerfile.write_text("FROM " + image + "\n")
        stage_file(dockerfile, root / "environment/Dockerfile")
        fail_closed = temp / "test.sh"
        fail_closed.write_text('#!/bin/sh\necho "BIABench requires maintained custom verifier" >&2\nexit 1\n')
        stage_file(fail_closed, root / "tests/test.sh")
        row = json.loads((assets / "benchmark.jsonl").read_text())
        dataset = temp / "benchmark.jsonl"
        dataset.write_text(json.dumps(row) + "\n")
        published = stage_file(dataset, target / "benchmark.jsonl")
        receipt = temp / "preparation.json"
        receipt.write_text(
            json.dumps(
                {
                    "image": image,
                    "prepare_sha256": _sha(Path(__file__)),
                    "source_manifest_sha256": _sha(Path(__file__).with_name("frozen_sources.json")),
                    "files": {name: _sha(target / name) for name in _published_names()},
                },
                indent=2,
            )
        )
        stage_file(receipt, target / "preparation.json")
        return published
