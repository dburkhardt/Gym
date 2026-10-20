# SPDX-License-Identifier: Apache-2.0
"""Publish preserved P-Bench Harbor tasks with explicit protocol provenance."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_gym.dataset_staging import stage_file


def prepare(source: Path, destination: Path, size: int = 5) -> Path:
    if size not in (5, 20, 50):
        raise ValueError("Expected frozen prefix5,20 or50")
    manifest = json.loads(Path(__file__).with_name("frozen_sources.json").read_text())
    selected = manifest["tasks"][:size]
    source = source.resolve(strict=True)
    destination = destination.parent.resolve() / destination.name
    # Validate all selected scientific files before publishing any task.
    for task in selected:
        for relative, expected in task["files"].items():
            path = source / "tasks" / task["harbor_task"] / relative
            if path.is_symlink() or not path.resolve().is_relative_to(source):
                raise ValueError("Unsafe P-Bench source path")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Changed frozen P-Bench file: {task['harbor_task']}/{relative}")
    for task in selected:
        base = destination / "tasks" / task["harbor_task"]
        for existing in base.rglob("*"):
            if existing.is_symlink():
                raise ValueError("P-Bench destination contains a symlink")
            if existing.is_file():
                relative = str(existing.relative_to(base))
                if existing.name.startswith(".") and existing.name.endswith(".stage.lock"):
                    relative = str(existing.relative_to(base).with_name(existing.name[1:-11]))
                if relative not in task["files"]:
                    raise ValueError("P-Bench destination contains unrecorded files")
        for relative, expected in task["files"].items():
            target = base / relative
            if any(p.is_symlink() for p in [destination, *target.parents] if p != p.parent):
                raise ValueError("P-Bench staging paths must not alias another directory")
            published = stage_file(source / "tasks" / task["harbor_task"] / relative, target, expected_sha256=expected)
            if relative.endswith(".sh"):
                published.chmod(0o755)
    rows = [
        {
            "task_id": "pbench::" + t["harbor_task"],
            "instance_id": "pbench::" + t["harbor_task"],
            "responses_create_params": {"input": [], "temperature": 0.0},
            "agent_ref": {"name": "harbor_agent"},
        }
        for t in selected
    ]
    with TemporaryDirectory() as directory:
        candidate = Path(directory) / "benchmark.jsonl"
        candidate.write_text("".join(json.dumps(row) + "\n" for row in rows))
        receipt = Path(directory) / "provenance.json"
        receipt.write_text(
            json.dumps(
                {
                    "dataset_revision": manifest["dataset_revision"],
                    "protocol_change": manifest["protocol_change"],
                    "dependency_caveat": manifest["dependency_caveat"],
                    "task_ids": [r["task_id"] for r in rows],
                    "dataset_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                    "status": "prepared_only; runtime image and isolated verifier qualification pending",
                },
                indent=2,
            )
            + "\n"
        )
        stage_file(receipt, destination / receipt.name)
        return stage_file(candidate, destination / candidate.name)


if __name__ == "__main__":
    print(
        prepare(
            Path(os.environ["PBENCH_PREPARED_SOURCE"]),
            Path(os.environ.get("PBENCH_DATA", str(Path(__file__).parent / "data"))),
            int(os.environ.get("PBENCH_SAMPLE_SIZE", "5")),
        )
    )
