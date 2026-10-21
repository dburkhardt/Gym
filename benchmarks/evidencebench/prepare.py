# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage frozen rows with a native task identity and provenance receipt."""

from pathlib import Path

from benchmarks.scientific_agent_campaign.stage_sample import prepare_sample


def prepare() -> Path:
    return prepare_sample(Path(__file__).resolve().parents[2] / "resources_servers/evidencebench/data/benchmark.jsonl")


def is_prepared_data_current(fpath: Path) -> bool:
    """Validate the selected frozen sample before native cached preparation."""
    from benchmarks.scientific_agent_campaign.stage_sample import is_sample_current

    return is_sample_current(
        fpath,
        default_destination=Path(__file__).resolve().parents[2]
        / "resources_servers/evidencebench/data/benchmark.jsonl",
    )


if __name__ == "__main__":
    print(prepare())
