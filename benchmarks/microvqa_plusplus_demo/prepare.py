# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve existing image-detail defaults while publishing atomically with provenance."""

from pathlib import Path

from benchmarks.scientific_agent_campaign.stage_sample import prepare_sample


def prepare() -> Path:
    return prepare_sample(
        Path(__file__).resolve().parents[2] / "resources_servers/microvqa_plusplus_demo/data/benchmark.jsonl",
        identity_metadata_key="id",
        normalize_image_detail=True,
    )


if __name__ == "__main__":
    print(prepare())
