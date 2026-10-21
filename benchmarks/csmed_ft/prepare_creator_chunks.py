# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native prepare-script entrypoint for explicitly configured creator chunks."""

import json
import os
import subprocess
from pathlib import Path

from benchmarks.csmed_ft.chunk_preparation import prepare


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[2] / "resources_servers/csmed_ft/data/creator_chunks.jsonl"
    recipe = Path(os.environ["CSMED_CREATOR_RECIPE"])
    interpreter = os.environ.get("CSMED_CREATOR_PYTHON")
    if interpreter:
        subprocess.run(
            [
                interpreter,
                "-m",
                "benchmarks.csmed_ft.chunk_preparation",
                "--recipe",
                str(recipe),
                "--output",
                str(output),
            ],
            check=True,
            timeout=300,
        )
    else:
        print(prepare(json.loads(recipe.read_text()), output=output))
