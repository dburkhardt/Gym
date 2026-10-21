# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish frozen creator questions without resampling or exposing references."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from nemo_gym.dataset_staging import stage_file


def prepare() -> Path:
    source = Path(os.environ["MOOSE_FROZEN_TASKS"])
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != os.environ["MOOSE_FROZEN_SHA256"]:
        raise ValueError("Frozen MOOSE-Chem source hash mismatch")
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if not rows or len(rows) > 50 or len(set(ids)) != len(ids):
        raise ValueError("Expected unique frozen MOOSE-Chem tasks")
    prepared = []
    for row in rows:
        if isinstance(row["id"], bool) or not isinstance(row["id"], int):
            raise ValueError("Creator task identity must be integer")
        prepared.append(
            {
                "task_id": str(row["id"]),
                "responses_create_params": {
                    "input": [{"role": "user", "content": row["question"] + "\n\n" + row["survey"]}]
                },
                "verifier_metadata": row,
            }
        )
    destination = Path(__file__).parent / "data/benchmark.jsonl"
    with tempfile.TemporaryDirectory() as temporary:
        candidate = Path(temporary) / "rows.jsonl"
        candidate.write_text("".join(json.dumps(row) + "\n" for row in prepared))
        output_digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        stage_file(candidate, destination, expected_sha256=output_digest)
        receipt = Path(temporary) / "source.json"
        receipt.write_text(
            json.dumps({"source_sha256": digest, "output_sha256": output_digest, "task_ids": ids}, indent=2)
        )
        stage_file(receipt, destination.with_suffix(".source.json"))
    return destination


if __name__ == "__main__":
    print(prepare())
