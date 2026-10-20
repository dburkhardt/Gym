# SPDX-License-Identifier: Apache-2.0
"""Preserve Corral's frozen native tasks; no flat-answer substitution."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_gym.dataset_staging import stage_file


def stage(source: Path, destination: Path) -> Path:
    manifest = json.loads(Path(__file__).with_name("frozen_sources.json").read_text())
    source, destination = source.resolve(), destination.resolve()
    for relative, sha in manifest["files"].items():
        path = source / relative
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError(f"Corral pinned source mismatch: {relative}")
    rows = []
    for task_id in manifest["frozen5"]:
        relative = f"tasks/retrosynthesis/environments/level_3/tasks_json/make_{task_id.split('_')[1]}.json"
        native = json.loads((source / relative).read_text())
        if len(native) != 1 or native[0]["id"] != task_id:
            raise ValueError("Frozen task identity mismatch")
        rows.append(
            {
                "task_id": task_id,
                "instance_id": task_id,
                "uuid": task_id,
                "responses_create_params": {"input": []},
                "verifier_metadata": {
                    "native_task": native[0],
                    "source_path": relative,
                    "source_sha256": manifest["files"][relative],
                },
            }
        )
    if len({r["task_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate frozen task IDs")
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temporary:
        prepared = Path(temporary) / "benchmark.jsonl"
        prepared.write_text("".join(json.dumps(row) + "\n" for row in rows))
        target = stage_file(prepared, destination / prepared.name)
        receipt = Path(temporary) / "benchmark.source.json"
        receipt.write_text(
            json.dumps(
                {
                    "source_commit": manifest["source_commit"],
                    "source_files": manifest["files"],
                    "task_ids": manifest["frozen5"],
                    "output_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest(),
                },
                indent=2,
            )
            + "\n"
        )
        stage_file(receipt, destination / receipt.name)
    return target


def prepare():
    return stage(Path(os.environ["CORRAL_SOURCE"]), Path(os.environ.get("CORRAL_DATA", "benchmarks/corral/data")))


if __name__ == "__main__":
    prepare()
