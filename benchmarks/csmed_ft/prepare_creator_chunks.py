# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native prepare-script entrypoint for explicitly configured creator chunks."""

import json
import os
from pathlib import Path

from benchmarks.csmed_ft.chunk_preparation import prepare


if __name__ == "__main__":
    print(
        prepare(
            json.loads(Path(os.environ["CSMED_CREATOR_RECIPE"]).read_text()),
            output=Path(__file__).resolve().parents[2] / "resources_servers/csmed_ft/data/creator_chunks.jsonl",
        )
    )
