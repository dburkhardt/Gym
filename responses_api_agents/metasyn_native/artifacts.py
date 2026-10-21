# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attempt-scoped creator artifacts; this does not implement agent recovery."""

import hashlib
import json
from pathlib import Path

from nemo_gym.base_responses_api_model import CaptureStore
from nemo_gym.episode_types import EpisodeId


def attempt_root(output_root: Path, review_id: int, lineage: str) -> Path:
    EpisodeId.from_capture_key(lineage)
    key = hashlib.sha256(json.dumps([review_id, lineage]).encode()).hexdigest()
    return output_root / "attempts" / str(review_id) / key


def assert_prior_attempts_settled(output_root: Path, review_id: int, lineage: str, capture_root: Path) -> None:
    """Do not turn a new attempt suffix into permission to resend unknown work."""
    requested = EpisodeId.from_capture_key(lineage)
    legacy = output_root / str(review_id)
    if legacy.exists():
        raise ValueError("Legacy task-only MetaSyn artifacts require explicit reconciliation")
    parent = output_root / "attempts" / str(review_id)
    if not parent.exists():
        return
    for attempt in parent.iterdir():
        out = attempt / str(review_id)
        identity = json.loads((out / "identity.json").read_text())
        previous_lineage = identity["lineage"]
        if identity["review_id"] != review_id or attempt != attempt_root(output_root, review_id, previous_lineage):
            raise ValueError("MetaSyn prior attempt identity does not match its artifact path")
        if EpisodeId.from_capture_key(previous_lineage).rollout_id != requested.rollout_id:
            continue  # An independent repeat is not a retry or reconciliation of this episode.
        if identity.get("capture_root") != str(capture_root.resolve()) or not capture_root.is_dir():
            raise ValueError("MetaSyn prior attempt capture store is missing or changed")
        CaptureStore(capture_root, provider_dispatch_guard=True).assert_dispatch_reconciled([previous_lineage])
        execution = json.loads((out / "execution.json").read_text())
        if execution.get("return_code") != 0 or execution.get("error_type") is not None:
            raise ValueError("MetaSyn prior creator execution requires reconciliation")
        if json.loads((out / "evaluation.json").read_text()).get("status") != "succeeded":
            raise ValueError("MetaSyn prior evaluation requires reconciliation")
