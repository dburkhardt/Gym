# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register all four released frozen tasks, retaining explicit ET blockers."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.ainsteinbench.native import IMAGES
from benchmarks.ainsteinbench.source import checked_source
from nemo_gym.dataset_staging import stage_file


def stage(source: Path, destination: Path):
    rows, hashes = checked_source(source)
    records = []
    with TemporaryDirectory() as temporary:
        temp = Path(temporary)
        for task_id, raw in rows.items():
            supported = raw["question_type"] == "multi_swe_bench"
            image = (
                IMAGES[int(raw["content"]["pr_number"])]
                if supported
                else "sha256:44e33aeed3ba691bcff99846141ffd329edf6b32060841b0f151d3867adc9df0"
            )
            # ET remains registered against its actual preserved image, but blocked.
            blocker = (
                None
                if supported
                else "ET gold native build timeout; reference/cache isolation and perturbation validity unresolved"
            )
            workdir = "/home/pyscf" if supported else raw["environment"]["working_directory"]
            records.append(
                {
                    "task_id": task_id,
                    "instance_id": task_id,
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
cpus = 1
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
                    "source_hashes": hashes,
                    "task_ids": list(rows),
                    "supported_native_tasks": list(rows)[:2],
                    "blocked_tasks": list(rows)[2:],
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
