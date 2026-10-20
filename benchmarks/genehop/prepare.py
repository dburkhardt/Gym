# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate already frozen creator rows; no acquisition, resampling or mutation."""

import json
import os
from pathlib import Path


def prepare() -> Path:
    """Return the configured frozen sample through Gym's preparation interface."""
    path = Path(os.environ["GENEHOP_SAMPLE"]).absolute()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("GeneHop sample is empty")
    for row in rows:
        expected = [{"role": "user", "content": row["verifier_metadata"]["question"]}]
        if row["responses_create_params"]["input"] != expected:
            raise ValueError("GeneHop solver prompt differs from the frozen creator question")
    from benchmarks.scientific_agent_campaign.stage_sample import prepare_sample

    destination = Path(os.environ.get("GENEHOP_PREPARED_SAMPLE", "benchmarks/genehop/data/benchmark.jsonl"))
    return prepare_sample(destination, source=path)


if __name__ == "__main__":
    prepare()
