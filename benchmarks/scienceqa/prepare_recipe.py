# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare an explicitly named creator-recipe dataset without changing legacy data."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from benchmarks.scienceqa.creator_recipe import prepare as prepare_packet
from nemo_gym.dataset_staging import stage_file


def prepare() -> Path:
    source = Path(os.environ["CAMPAIGN_SAMPLE_JSONL"])
    content = source.read_bytes()
    source_hash = hashlib.sha256(content).hexdigest()
    expected = os.environ.get("CAMPAIGN_SAMPLE_SHA256")
    if expected and expected != source_hash:
        raise ValueError("Frozen sample hash differs")
    sample = [json.loads(line) for line in content.splitlines() if line.strip()]
    ids = [str(row.get("task_id", row.get("uuid"))) for row in sample]
    packet = prepare_packet(Path(os.environ["SCIENCEQA_CREATOR_REPO"]), ids)
    rows = []
    for request, reference in zip(packet["solver_requests"], packet["evaluator_references"]):
        rows.append(
            {
                "task_id": request["task_id"],
                "uuid": request["task_id"],
                "responses_create_params": {"input": [{"role": "user", "content": json.dumps(request["request"])}]},
                "verifier_metadata": reference | {"protocol": packet["protocol"]},
            }
        )
    output = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    destination = Path(__file__).parent / "data" / "creator-recipe.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        candidate = Path(temporary) / "rows.jsonl"
        candidate.write_bytes(output)
        stage_file(candidate, destination, expected_sha256=hashlib.sha256(output).hexdigest())
        candidate.write_text(json.dumps(packet, indent=2) + "\n")
        stage_file(candidate, destination.with_suffix(".packet.json"))
        candidate.write_text(
            json.dumps(
                {
                    "source_sha256": source_hash,
                    "output_sha256": hashlib.sha256(output).hexdigest(),
                    "task_ids": ids,
                    "protocol": packet["protocol"],
                    "source_hashes": packet["source_hashes"],
                },
                indent=2,
            )
            + "\n"
        )
        stage_file(candidate, destination.with_suffix(".source.json"))
    return destination


if __name__ == "__main__":
    print(prepare())
