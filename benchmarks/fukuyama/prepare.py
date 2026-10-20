# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import shutil
from benchmarks.scientific_agent_campaign.use_sample import prepare as validate_sample

def prepare() -> Path:
    source = validate_sample()
    destination = Path(__file__).parent / "data" / "benchmark.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source != destination.resolve():
        shutil.copyfile(source, destination)
    return destination.resolve()

if __name__ == "__main__":
    print(prepare())
