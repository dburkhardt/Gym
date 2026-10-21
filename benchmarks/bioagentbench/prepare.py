# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish all ten frozen native workflow prompts, without scorer truth."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from nemo_gym.dataset_staging import stage_file


TASKS = [
    "comparative-genomics",
    "cystic-fibrosis",
    "evolution",
    "transcript-quant",
    "metagenomics",
    "alzheimer-mouse",
    "viral-metagenomics",
    "deseq",
    "giab",
    "single-cell",
]


def rows(metadata: Path) -> list[dict]:
    expected = json.loads(Path(__file__).with_name("source-sha256.json").read_text())["metadata_sha256"]
    if hashlib.sha256(metadata.read_bytes()).hexdigest() != expected:
        raise ValueError("Frozen metadata hash changed")
    source = {row["task_id"]: row for row in json.loads(metadata.read_text())}
    if set(source) != set(TASKS):
        raise ValueError("Full frozen ten-workflow population required")
    output = []
    for task_id in TASKS:
        row = source[task_id]
        reference = bool(row["download_urls"]["reference_data"])
        prompt = (
            row["task_prompt"].strip()
            + "\n\nUse the files in `data/`"
            + (" and `reference/`" if reference else "")
            + ". Write the requested final deliverable to `results/`."
        )
        output.append(
            {
                "task_id": task_id,
                "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
                "verifier_metadata": {"id": task_id, "metadata_sha256": expected},
            }
        )
    return output


def prepare() -> Path:
    prepared = rows(Path(os.environ["BIOAGENT_METADATA"]))
    destination = Path(__file__).parent / "data/benchmark.jsonl"
    with tempfile.TemporaryDirectory() as temporary:
        candidate = Path(temporary) / "benchmark.jsonl"
        candidate.write_text("".join(json.dumps(row) + "\n" for row in prepared))
        stage_file(candidate, destination, expected_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
    return destination


if __name__ == "__main__":
    print(prepare())
