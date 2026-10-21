# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned preparation; download only task definitions here, never reference answers into prompts."""

import hashlib
import os
import json
from pathlib import Path


DATA_REVISION = "078420b69e25c41c99a79135724b6ae6a6c82b69"
DATA_REPO = "promptbio-ai/promptbio-bench-data"
SCORER_REVISION = "e2a1947894eac530900247f00c0b726363b540ed"


FROZEN_SELECTION_SHA256 = "6432f4c88a94e4f5c14ae9aa0ccfecbf61a6e01cf947ef561e99fdc2cc5edd01"


def prepare_frozen(snapshot: Path, selection: Path, output: Path) -> Path:
    """Recover the exact frozen sample from checksum-verified cached task definitions."""
    from tempfile import TemporaryDirectory
    from nemo_gym.dataset_staging import stage_file

    selected_bytes = selection.read_bytes()
    if hashlib.sha256(selected_bytes).hexdigest() != FROZEN_SELECTION_SHA256:
        raise ValueError("Frozen PromptBio selection changed")
    selected = json.loads(selected_bytes)
    rows = []
    for path in sorted(snapshot.glob("tasks/*/task.json")):
        task = json.loads(path.read_text())
        rows.append({"responses_create_params": {"input": [{"role": "user", "content": json.dumps(task)}]},
                     "verifier_metadata": {"task_id": task["id"]}})
    original = "".join(json.dumps(row) + "\n" for row in rows).encode()
    if len(rows) != 244 or hashlib.sha256(original).hexdigest() != selected["source_sha256"]:
        raise ValueError("Pinned PromptBio source definitions mismatch")
    by_id = {row["verifier_metadata"]["task_id"]: row for row in rows}
    frozen = [{**by_id[task_id], "task_id": task_id} for task_id in selected["task_ids"]]
    with TemporaryDirectory() as directory:
        candidate = Path(directory) / "tasks.jsonl"
        candidate.write_text("".join(json.dumps(row) + "\n" for row in frozen))
        return stage_file(candidate, output)


def prepare() -> Path:
    """Prepare all released task definitions, retaining authoritative IDs."""
    if os.environ.get("PROMPTBIO_FROZEN_SELECTION"):
        return prepare_frozen(Path(os.environ["PROMPTBIO_TASK_SNAPSHOT"]),
                              Path(os.environ["PROMPTBIO_FROZEN_SELECTION"]),
                              Path(__file__).parent / "data/promptbio_benchmark.jsonl")
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
    return output


if __name__ == "__main__":
    print(prepare())
