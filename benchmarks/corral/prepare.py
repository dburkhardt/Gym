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
                "agent_ref": {"name": "harbor_agent"},
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
    for task_id in manifest["frozen5"]:
        task = destination / "tasks" / task_id
        with TemporaryDirectory() as temporary:
            for name, text in {
                "instruction.md": task_id + "\n",
                "tests/test.sh": "#!/bin/sh\necho Native_Corral_custom_verifier_required >&2\nexit 1\n",
                "environment/Dockerfile": "FROM sha256:24b49fd1a5fd0bd03192106d6c50b44564d5f413ff69bb6333a58ed0d79a51e3\nRUN mkdir -p /workspace\n",
                "task.toml": 'version = "1.0"\nartifacts = ["/workspace"]\n[agent]\ntimeout_sec = 5040\n[verifier]\ntimeout_sec = 5040\n[environment]\ndocker_image = "sha256:24b49fd1a5fd0bd03192106d6c50b44564d5f413ff69bb6333a58ed0d79a51e3"\ncpus = 1\nmemory_mb = 128\nstorage_mb = 1024\nallow_internet = false\n',
            }.items():
                file = Path(temporary) / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(text)
                stage_file(file, task / name)
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
