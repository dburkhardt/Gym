# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declare operator-staged cases without publishing gated prompts or secrets."""

import hashlib
import json
from pathlib import Path


def prepare() -> Path:
    """Validate pinned local case manifests and write Gym rows in configured order."""
    directory = Path(__file__).resolve().parent
    config = json.loads((directory / "local-preparation.json").read_text())
    root = Path(config["settings_root"]).resolve(strict=True)
    rows = []
    seen = set()
    for entry in config["cases"]:
        path = (root / entry["settings_file"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Settings file escapes operator root")
        settings = json.loads(path.read_text())
        case_path = Path(settings["run_root"]) / "dataset" / "case.json"
        payload = case_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["case_sha256"]:
            raise ValueError("Staged case checksum mismatch")
        task_id = json.loads(payload)["task_id"]
        if task_id != settings["task_id"] or task_id in seen:
            raise ValueError("Mismatched or duplicate task ID")
        seen.add(task_id)
        rows.append(
            {"task_id": task_id, "settings_file": entry["settings_file"], "responses_create_params": {"input": []}}
        )
    if not rows:
        raise ValueError("No authorized staged cases configured")
    output = directory / "data" / "benchmark.jsonl"
    output.parent.mkdir(exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return output


if __name__ == "__main__":
    print(prepare())
