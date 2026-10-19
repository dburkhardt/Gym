# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned preparation; download only task definitions here, never reference answers into prompts."""

import json
from pathlib import Path


DATA_REVISION = "078420b69e25c41c99a79135724b6ae6a6c82b69"
DATA_REPO = "promptbio-ai/promptbio-bench-data"
SCORER_REVISION = "e2a1947894eac530900247f00c0b726363b540ed"


def prepare() -> str:
    """Prepare all released task definitions, retaining authoritative IDs."""
    from huggingface_hub import HfApi, hf_hub_download

    output = Path(__file__).parent / "data/promptbio_benchmark.jsonl"
    paths = sorted(
        p
        for p in HfApi().list_repo_files(DATA_REPO, repo_type="dataset", revision=DATA_REVISION)
        if p.startswith("tasks/") and p.endswith("/task.json")
    )
    rows = []
    for path in paths:
        task = json.loads(
            Path(hf_hub_download(DATA_REPO, path, repo_type="dataset", revision=DATA_REVISION)).read_text()
        )
        rows.append(
            {
                "responses_create_params": {"input": [{"role": "user", "content": json.dumps(task)}]},
                "verifier_metadata": {"task_id": task["id"]},
            }
        )
    if len(rows) != 244:
        raise ValueError(f"Expected 244 published tasks, found {len(rows)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(output)


if __name__ == "__main__":
    print(prepare())
