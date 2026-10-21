# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native attempt directories, without implicit retry or scientific recovery."""

import hashlib
import json
from pathlib import Path

from nemo_gym._checkpoint.evaluation import capture_snapshot
from nemo_gym.episode_types import EpisodeId


def attempt_path(root: Path, task_id: int, capture_key: str) -> Path:
    if type(task_id) is not int or task_id < 0:
        raise ValueError("Invalid native MOOSE task identifier")
    EpisodeId.from_capture_key(capture_key)
    digest = hashlib.sha256(json.dumps([task_id, capture_key]).encode()).hexdigest()
    return root / "attempts" / f"task{task_id}" / digest


def admit_attempt(root: Path, task_id: int, capture_key: str, capture_root: Path) -> tuple[Path, dict]:
    out = attempt_path(root, task_id, capture_key)
    if out.exists():
        raise FileExistsError("MOOSE episode already has artifacts; reconcile without rerunning")
    if (root / f"task{task_id}").exists():
        raise ValueError("Legacy task-only MOOSE artifacts require explicit reconciliation")
    episode = EpisodeId.from_capture_key(capture_key)
    snapshot = capture_snapshot([capture_root], [episode])
    store_identity = snapshot[str(capture_root.resolve())]["identity"]
    if out.parent.exists():
        for previous in out.parent.iterdir():
            identity = json.loads((previous / "execution-identity.jsonl").read_text().splitlines()[0])
            key = identity["capture_key"]
            if identity["task"] != task_id or previous != attempt_path(root, task_id, key):
                raise ValueError("MOOSE prior artifact identity mismatch")
            if EpisodeId.from_capture_key(key).rollout_id != episode.rollout_id:
                continue
            if (
                identity.get("capture_root") != str(capture_root.resolve())
                or identity.get("capture_store_identity") != store_identity
            ):
                raise ValueError("MOOSE original capture store changed")
            terminal = json.loads((previous / "native-execution.jsonl").read_text().splitlines()[-1])
            if terminal.get("return_code") != 0 or terminal.get("error_type") is not None:
                raise ValueError("MOOSE prior creator execution requires reconciliation")
            # Creator failure/partial output cannot silently become a new attempt.
            if not all(
                (previous / name).is_file() for name in ("result.json", "screening.json", "model-traces.jsonl")
            ):
                raise ValueError("MOOSE prior creator execution requires reconciliation")
    return out, {
        "capture_key": capture_key,
        "episode": episode.model_dump(),
        "capture_root": str(capture_root.resolve()),
        "capture_store_identity": store_identity,
    }
