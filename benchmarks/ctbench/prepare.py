# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage frozen rows with a native task identity and provenance receipt."""

from pathlib import Path

from benchmarks.scientific_agent_campaign.stage_sample import prepare_sample


def prepare() -> Path:
    return prepare_sample(Path(__file__).resolve().parents[2] / "resources_servers/ctbench/data/benchmark.jsonl")


if __name__ == "__main__":
    print(prepare())
