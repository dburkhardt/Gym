# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy audited campaign rows, never download an unpinned dataset implicitly."""

import os
import json

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from pathlib import Path


if __name__ == "__main__":
    source = Path(os.environ["CAMPAIGN_SAMPLE_JSONL"])
    destination = Path(__file__).resolve().parents[2] / "resources_servers/microvqa_plusplus_demo/data/benchmark.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for row in rows:
        for message in row["responses_create_params"]["input"]:
            if isinstance(message["content"], list):
                for part in message["content"]:
                    if part["type"] == "input_image":
                        part.setdefault("detail", "auto")
        NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
    destination.write_text("".join(json.dumps(row) + "\n" for row in rows))
