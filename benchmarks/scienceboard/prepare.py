# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage the creator's exact desktop parity selection, without claiming a runnable agent."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_gym.dataset_staging import stage_file


MANIFEST = Path(__file__).with_name("creator-parity20.json")


def checked_source(root: Path, relative: str, expected: str) -> Path:
    """Require a regular, contained source file with the pinned creator bytes."""
    root = root.absolute()
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Invalid creator source path")
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent == root or root in parent.parents):
        raise ValueError("Creator source symlink is unsupported")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Creator source file is missing or escapes source")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Creator source changed: {relative}")
    return path


def prepare(source: Path | None = None, destination: Path | None = None) -> Path:
    """Publish exact task bytes separately from reference-free native input rows."""
    source = source or Path(os.environ["SCIENCEBOARD_SOURCE_ROOT"])
    destination = destination or Path(__file__).parent / "data"
    manifest = json.loads(MANIFEST.read_text())
    for relative, expected in manifest["files"].items():
        checked_source(source, relative, expected)
    rows = []
    tasks = []
    for item in manifest["tasks"]:
        path = checked_source(source, item["path"], item["sha256"])
        task = json.loads(path.read_text())
        if task["sort"] != "VM" or task.get("snapshot", "sci_bench") != "sci_bench":
            raise ValueError("Only exact creator desktop protocol is supported")
        tasks.append((path, item))
        rows.append(
            {
                "task_id": item["task_id"],
                "responses_create_params": {"input": [{"role": "user", "content": task["instruction"]}]},
                "verifier_metadata": {
                    "task_id": item["task_id"],
                    "task_sha256": item["sha256"],
                    "creator_task_path": item["path"],
                    "source_revision": manifest["source_revision"],
                },
            }
        )
    # Validate all inputs before publishing. These trusted host-side JSON files include
    # verifier selectors; never mount this task directory into the desktop sandbox.
    for path, item in tasks:
        stage_file(path, destination / "creator_tasks" / item["path"], expected_sha256=item["sha256"])
    stage_file(MANIFEST, destination / "selection.json")
    with TemporaryDirectory() as tmp:
        candidate = Path(tmp) / "benchmark.jsonl"
        candidate.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return stage_file(candidate, destination / "benchmark.jsonl")


if __name__ == "__main__":
    print(prepare())
