# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind preserved MDArena samples without altering scientific task files."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from nemo_gym.dataset_staging import stage_file


def stage(source: Path, destination: Path) -> Path:
    """Validate a frozen 5/20/50 sample and add only Gym's stable identity alias."""
    content = source.read_bytes()
    source_hash = hashlib.sha256(content).hexdigest()
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    frozen = json.loads(Path(__file__).with_name("frozen-samples.json").read_text())
    expected = frozen.get(str(len(rows)))
    if expected is None or source_hash != expected["sha256"]:
        raise ValueError("MDArena sample must match preserved frozen5/20/50 source bytes")
    identities = [row.get("instance_id") for row in rows]
    if identities != expected["instance_ids"] or len(set(identities)) != len(identities):
        raise ValueError("MDArena frozen identities/order mismatch or duplicates")
    for row in rows:
        if row.get("agent_ref") != {"name": "harbor_agent"}:
            raise ValueError("MDArena frozen agent routing changed")
        if row.get("responses_create_params") != {"input": [], "temperature": 0.0}:
            raise ValueError("MDArena frozen request parameters changed")
        if "task_id" in row and row["task_id"] != row["instance_id"]:
            raise ValueError("MDArena task identity mismatch")
        row["task_id"] = row["instance_id"]
    output = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
    output_hash = hashlib.sha256(output).hexdigest()
    destination = destination.parent.resolve() / destination.name
    receipt = {
        "source_commit": "d1d789d92a9935e716182e31d99393edcd4ac7d6",
        "source_sha256": source_hash,
        "output_sha256": output_hash,
        "task_ids": identities,
        "rows": len(rows),
        "normalizations": ["task_id aliases unchanged instance_id"],
        "scientific_task_files_modified": False,
    }
    with tempfile.TemporaryDirectory(prefix="gym-mdarena-sample-") as temporary:
        candidate = Path(temporary) / "benchmark.jsonl"
        candidate.write_bytes(output)
        stage_file(candidate, destination, expected_sha256=output_hash)
        proof = Path(temporary) / "provenance.json"
        proof.write_text(json.dumps(receipt, indent=2) + "\n")
        stage_file(proof, destination.with_suffix(".source.json"))
    return destination


def prepare() -> Path:
    """Prepare the explicitly selected preserved sample for native Gym planning."""
    return stage(
        Path(os.environ["MDARENA_FROZEN_SAMPLE"]),
        Path(os.environ.get("MDARENA_PREPARED_SAMPLE", "benchmarks/mdarena/data/benchmark.jsonl")),
    )


if __name__ == "__main__":
    print(prepare())
