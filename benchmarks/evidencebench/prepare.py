# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy audited campaign rows, never download an unpinned dataset implicitly."""

import os
from pathlib import Path

from nemo_gym.dataset_staging import stage_file


if __name__ == "__main__":
    source = Path(os.environ["CAMPAIGN_SAMPLE_JSONL"])
    destination = Path(__file__).resolve().parents[2] / "resources_servers/evidencebench/data/benchmark.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage_file(source, destination)
