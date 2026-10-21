# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map creator artifacts without excluding genuine model failures or inventing scores."""

import hashlib
from pathlib import Path


def read_creator_result(
    result_path: Path, *, task_id: str, transport_outcome: str = "not_dispatched", creator_error: str | None = None
) -> dict:
    """Retain model failures in attempted denominator; absent scientific score remains null."""
    model_failure = transport_outcome in {"model_output_budget", "model_invalid_action"}
    unresolved = transport_outcome == "transport_unresolved"
    record = {
        "task_id": task_id,
        "raw_reward": None,
        "attempted": transport_outcome != "not_dispatched",
        "scored": False,
        "mask_sample": True,
        "transport_outcome": transport_outcome,
        "measurement_review": {"status": "unreviewed"},
    }
    if result_path.is_symlink() or any(p.is_symlink() for p in result_path.parents):
        return {**record, "disposition": "verifier_artifact_failure", "reason": "Result artifact symlink refused"}
    if not result_path.exists():
        return {
            **record,
            "mask_sample": not model_failure or creator_error is not None,
            "disposition": transport_outcome if model_failure or unresolved else "creator_result_missing",
            "reason": creator_error or "No result.out produced; no scientific score fabricated",
        }
    data = result_path.read_bytes()
    record["result_sha256"] = hashlib.sha256(data).hexdigest()
    try:
        value = data.decode().strip()
    except UnicodeDecodeError:
        value = None
    if value not in {"0", "1"}:
        return {
            **record,
            "disposition": "verifier_artifact_failure",
            "reason": "Malformed native result.out retained unchanged",
        }
    return {
        **record,
        "raw_reward": int(value),
        "scored": True,
        "mask_sample": unresolved or creator_error is not None,
        "disposition": transport_outcome
        if model_failure or unresolved
        else ("execution_failure_with_raw_result" if creator_error else "creator_result_unreviewed"),
        "reason": creator_error,
    }
