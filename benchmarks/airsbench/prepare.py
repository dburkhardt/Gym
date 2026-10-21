# SPDX-License-Identifier: Apache-2.0
"""Validate and publish complete preserved AIRS scientific tasks, never metadata-only caches."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_gym.dataset_staging import stage_file


def prepare(roots: dict[str, str], destination: Path) -> Path:
    manifest = json.loads(Path(__file__).with_name("frozen_sources.json").read_text())
    tasks = manifest["tasks"]
    if set(roots) != {task["name"] for task in tasks}:
        raise ValueError("Provide all five frozen research task roots; do not resample QM9 rows as tasks")
    # Validate every source before publishing any task. Historical laptop caches
    # can contain Arrow state metadata without the actual prediction/label arrays.
    for task in tasks:
        root = Path(roots[task["name"]]).resolve(strict=True)
        for relative, expected in task["files"].items():
            source = root / relative
            if source.is_symlink() or not source.resolve().is_relative_to(root) or not source.is_file():
                raise ValueError(f"Missing or unsafe scientific asset: {task['name']}/{relative}")
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Changed scientific asset: {task['name']}/{relative}")
    destination = destination.parent.resolve() / destination.name
    if destination.is_symlink():
        raise ValueError("Destination cannot be a symlink")
    for task in tasks:
        root = Path(roots[task["name"]]).resolve(strict=True)
        target = destination / "tasks" / task["name"]
        for path in target.rglob("*"):
            if path.is_symlink() or (
                path.is_file()
                and str(path.relative_to(target)) not in task["files"]
                and not path.name.endswith(".stage.lock")
            ):
                raise ValueError("Destination contains unrecorded task state")
        for relative, expected in task["files"].items():
            path = target / relative
            if any(parent.is_symlink() for parent in path.parents):
                raise ValueError("Destination parent cannot be a symlink")
            published = stage_file(root / relative, path, expected_sha256=expected)
            if relative.endswith(".sh"):
                published.chmod(0o755)
    rows = [
        {
            "task_id": "airsbench::" + t["name"],
            "instance_id": "airsbench::" + t["name"],
            "responses_create_params": {"input": [], "temperature": 0.0},
            "agent_ref": {"name": "harbor_agent"},
        }
        for t in tasks
    ]
    with TemporaryDirectory() as temporary:
        candidate = Path(temporary) / "benchmark.jsonl"
        candidate.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return stage_file(candidate, destination / "benchmark.jsonl")


if __name__ == "__main__":
    print(prepare(json.loads(os.environ["AIRS_TASK_ROOTS_JSON"]), Path(__file__).parent / "data"))
