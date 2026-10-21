# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize the one preserved CPU task without exposing verifier assets to Harbor."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.biabench.prepare import stage
from nemo_gym.dataset_staging import stage_file


SOLVER_IMAGE = "sha256:43e4fe86544cd2a980b0d7d9c10421d4458673a298203dc40aee30d324cfcf78"
TASK = "wound-healing-speed-kymograph-gigadb100118"


def prepare() -> Path:
    source, task, assets = (Path(os.environ[k]) for k in ("BIABENCH_SOURCE", "BIABENCH_TASK", "BIABENCH_DATA"))
    stage(source, task, assets)
    target = Path(__file__).parent / "data"
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
            + SOLVER_IMAGE
            + """"
cpus = 4
memory_mb = 8192
storage_mb = 10240
allow_internet = false
"""
        )
        stage_file(definition, root / "task.toml")
        dockerfile = temp / "Dockerfile"
        dockerfile.write_text("FROM " + SOLVER_IMAGE + "\n")
        stage_file(dockerfile, root / "environment/Dockerfile")
        fail_closed = temp / "test.sh"
        fail_closed.write_text('#!/bin/sh\necho "BIABench requires maintained custom verifier" >&2\nexit 1\n')
        stage_file(fail_closed, root / "tests/test.sh")
        row = json.loads((assets / "benchmark.jsonl").read_text())
        dataset = temp / "benchmark.jsonl"
        dataset.write_text(json.dumps(row) + "\n")
        return stage_file(dataset, target / "benchmark.jsonl")
