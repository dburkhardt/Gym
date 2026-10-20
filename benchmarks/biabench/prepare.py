# SPDX-License-Identifier: Apache-2.0
"""Validate pinned BIABench assets and publish native task identity, without inference."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_gym.dataset_staging import stage_file


def validate(root: Path, files: dict[str, str]) -> None:
    root = root.resolve()
    for relative, expected in files.items():
        path = root / relative
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError(f"Unsafe pinned asset: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Pinned BIABench asset mismatch: {relative}")


def stage(source: Path, task: Path, destination: Path) -> Path:
    manifest = json.loads(Path(__file__).with_name("frozen_sources.json").read_text())
    validate(source, manifest["creator_files"])
    validate(task, manifest["task_files"])
    row = {
        "task_id": manifest["task_id"],
        "instance_id": "biabench::" + manifest["task_id"],
        "responses_create_params": {"input": []},
        "verifier_metadata": {
            "tier": manifest["tier"],
            "outcome_scorer": manifest["outcome_scorer"],
            "process_score_required": True,
        },
    }
    # No task assets or references are exposed to a solver by this identity stage.
    # The eventual managed adapter must provide separately validated mounts.
    with TemporaryDirectory() as directory:
        temporary = Path(directory)
        data = temporary / "benchmark.jsonl"
        data.write_text(json.dumps(row) + "\n")
        target = stage_file(data, destination / data.name)
        proof = temporary / "benchmark.source.json"
        proof.write_text(
            json.dumps(manifest | {"output_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}, indent=2) + "\n"
        )
        stage_file(proof, destination / proof.name)
    return target


def prepare() -> Path:
    return stage(
        Path(os.environ["BIABENCH_SOURCE"]),
        Path(os.environ["BIABENCH_TASK"]),
        Path(os.environ.get("BIABENCH_DATA", "benchmarks/biabench/data")),
    )


if __name__ == "__main__":
    print(prepare())
