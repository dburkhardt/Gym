# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate pinned creator tasks and preserved overlays, then stage native Harbor inputs."""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_gym.dataset_staging import stage_file


SOURCE_COMMIT = "772d3f3e1a93b875f49ddb692e6dfc3db1d3593d"
MANIFEST_SHA256 = "387ff4e1750b432906d555576a55a7d31dd859a2efbf47c018967d1ea834fa5a"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(checkout: Path, prepared: Path) -> dict:
    """Refuse changed scientific inputs or unrecorded task overlay files before staging."""
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if head != SOURCE_COMMIT:
        raise ValueError("CompChemBench source revision mismatch")
    manifest_path = prepared / "manifest.json"
    if digest(manifest_path) != MANIFEST_SHA256:
        raise ValueError("CompChemBench overlay manifest mismatch")
    manifest = json.loads(manifest_path.read_text())
    for task in manifest["tasks"]:
        name = task["task"]
        source = checkout / "tasks" / name
        target = prepared / "tasks" / name
        expected = dict(task["source_files"])
        for relative, checksum in expected.items():
            if digest(source / relative) != checksum:
                raise ValueError(f"Changed creator file: {name}/{relative}")
        expected.update(
            {
                "task.toml": task["prepared_config_sha256"],
                "environment/Dockerfile": task["prepared_dockerfile_sha256"],
                "tests/test.sh": task["prepared_test_sha256"],
                "tests/capture_artifacts.py": task["artifact_capture_sha256"],
            }
        )
        actual = {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()}
        if actual != set(expected):
            raise ValueError(f"Unrecorded or missing prepared files: {name}")
        for relative, checksum in expected.items():
            path = target / relative
            if path.is_symlink() or digest(path) != checksum:
                raise ValueError(f"Changed prepared file: {name}/{relative}")
        task["validated_files"] = expected
    return manifest


def stage(checkout: Path, prepared: Path, destination: Path, *, limit: int = 5) -> Path:
    """Keep frozen manifest order and publish the dataset only after every selected task validates."""
    manifest = validate(checkout, prepared)
    if limit not in (5, 20, 43):
        raise ValueError("Supported frozen prefixes are 5, 20 and all 43 tasks")
    selected = manifest["tasks"][:limit]
    destination.mkdir(parents=True, exist_ok=True)
    for task in selected:
        for relative, checksum in task["validated_files"].items():
            src = prepared / "tasks" / task["task"] / relative
            dst = destination / "tasks" / task["task"] / relative
            stage_file(src, dst, expected_sha256=checksum)
            dst.chmod(src.stat().st_mode & 0o777)
    rows = [
        {
            "instance_id": f"compchem::{t['task']}",
            "task_id": f"compchem::{t['task']}",
            "responses_create_params": {"input": [], "temperature": 0.0},
            "agent_ref": {"name": "harbor_agent"},
        }
        for t in selected
    ]
    output = destination / "benchmark.jsonl"
    receipt = {
        "source_commit": SOURCE_COMMIT,
        "overlay_manifest_sha256": MANIFEST_SHA256,
        "rows": len(rows),
        "task_ids": [r["task_id"] for r in rows],
        "tasks": selected,
        "scientific_qualification": "Raw creator tests; not scientifically certified outcomes",
    }
    with TemporaryDirectory() as temp:
        temporary = Path(temp)
        staged = temporary / "benchmark.jsonl"
        staged.write_text("".join(json.dumps(r) + "\n" for r in rows))
        receipt["dataset_sha256"] = digest(staged)
        proof = temporary / "provenance.json"
        proof.write_text(json.dumps(receipt, indent=2) + "\n")
        stage_file(proof, destination / "provenance.json")
        stage_file(staged, output, expected_sha256=receipt["dataset_sha256"])
    return output


def prepare() -> Path:
    return stage(
        Path(os.environ["COMPCHEM_SOURCE_DIR"]),
        Path(os.environ["COMPCHEM_PREPARED_DIR"]),
        Path(__file__).parent / "data",
        limit=int(os.environ.get("COMPCHEM_LIMIT", "5")),
    )


if __name__ == "__main__":
    print(prepare())
