# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register frozen tasks with task-specific protocol and isolation readiness."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.ainsteinbench.et_native import DESITTER
from benchmarks.ainsteinbench.et_native import IMAGE as ET_IMAGE
from benchmarks.ainsteinbench.native import IMAGES
from benchmarks.ainsteinbench.source import checked_source
from nemo_gym.dataset_staging import stage_file


def stage(source: Path, destination: Path):
    rows, hashes = checked_source(source)
    existing = destination / "benchmark.jsonl"
    if existing.exists():
        previous = [json.loads(line) for line in existing.read_text().splitlines() if line.strip()]
        if any(row.get("instance_id") != f"ainsteinbench::{row.get('task_id')}" for row in previous):
            raise ValueError("Legacy AInstein routing artifact: prepare into a fresh versioned destination")
    provenance_path = destination / "provenance.json"
    if existing.exists() and (
        not provenance_path.exists()
        or json.loads(provenance_path.read_text()).get("preparation_version") != "et-native-desitter-v1"
    ):
        raise ValueError("Changed protocol preparation requires a fresh versioned destination")
    records = []
    with TemporaryDirectory() as temporary:
        temp = Path(temporary)
        for task_id, raw in rows.items():
            msb = raw["question_type"] == "multi_swe_bench"
            supported = msb or task_id == DESITTER
            image = IMAGES[int(raw["content"]["pr_number"])] if msb else ET_IMAGE
            # GRHydro remains registered but blocked: this image only qualifies deSitter isolation.
            blocker = (
                None
                if supported
                else "No task-specific isolated runtime qualification; GRHydro source remains visible"
            )
            workdir = "/home/pyscf" if msb else raw["environment"]["working_directory"]
            records.append(
                {
                    "task_id": task_id,
                    "instance_id": f"ainsteinbench::{task_id}",
                    "uuid": task_id,
                    "agent_ref": {"name": "harbor_agent"},
                    "responses_create_params": {"input": []},
                    "verifier_metadata": {
                        "question_type": raw["question_type"],
                        "blocked_reason": blocker,
                        "source_task_id": task_id,
                    },
                }
            )
            files = {
                "instruction.md": task_id + "\n",
                "task.toml": f'''version = "1.0"
artifacts = ["/submission"]
[agent]
timeout_sec = 18600
[verifier]
timeout_sec = 3600
environment_mode = "separate"
[environment]
docker_image = "{image}"
workdir = "{workdir}"
cpus = {1 if msb else 2}
memory_mb = 4096
storage_mb = 10240
network_mode = "no-network"
''',
                "tests/test.sh": "#!/bin/sh\necho AInstein_custom_native_verifier_required >&2\nexit 1\n",
            }
            for name, text in files.items():
                path = temp / task_id / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                stage_file(path, destination / "tasks" / task_id / name)
        data = temp / "benchmark.jsonl"
        data.write_text("".join(json.dumps(r) + "\n" for r in records))
        out = stage_file(data, destination / "benchmark.jsonl")
        proof = temp / "provenance.json"
        proof.write_text(
            json.dumps(
                {
                    "source_commit": "d9b1383e86c2ae43dcb3ddbcaf34c21ceb786cca",
                    "preparation_version": "et-native-desitter-v1",
                    "source_hashes": hashes,
                    "task_ids": list(rows),
                    "supported_native_tasks": [
                        r["task_id"] for r in records if r["verifier_metadata"]["blocked_reason"] is None
                    ],
                    "blocked_tasks": [
                        r["task_id"] for r in records if r["verifier_metadata"]["blocked_reason"] is not None
                    ],
                    "et_resource_variant": "Explicit 2 CPU / 4 GiB placement; creator specifies no caps; original command timeouts unchanged",
                    "qualification": "Terminus-independent creator four-tool async port; actual runtime gate required",
                },
                indent=2,
            )
            + "\n"
        )
        stage_file(proof, destination / "provenance.json")
    return out


def prepare():
    return stage(
        Path(os.environ["AINSTEIN_SOURCE"]), Path(os.environ.get("AINSTEIN_DATA", "benchmarks/ainsteinbench/data"))
    )


if __name__ == "__main__":
    print(prepare())
