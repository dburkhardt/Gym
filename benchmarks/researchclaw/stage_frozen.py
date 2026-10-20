# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish frozen Harbor request rows verbatim, not a replacement task prompt."""

import hashlib
import json
import os
from pathlib import Path

from nemo_gym.dataset_staging import stage_file


def prepare() -> Path:
    source = Path(os.environ["CAMPAIGN_SAMPLE_JSONL"])
    expected = os.environ["CAMPAIGN_SAMPLE_SHA256"]
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    ids = [row["instance_id"] for row in rows]
    if not rows or set(ids) != {"researchclaw::Neuroscience_000", "researchclaw::Life_002"} or len(ids) != 2:
        raise ValueError("Expected exact frozen ResearchClaw two-task calibration")
    if any(row["responses_create_params"]["input"] != [] for row in rows):
        raise ValueError("Harbor owns the frozen task prompt; request input must remain empty")
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("Frozen ResearchClaw request hash mismatch")
    destination = Path(__file__).parent / "data/benchmark.jsonl"
    stage_file(source, destination, expected_sha256=expected)
    return destination


if __name__ == "__main__":
    print(prepare())
