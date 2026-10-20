# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable task state; uncertain external effects require explicit human review."""

import json
import os
from pathlib import Path


def save_checkpoint(path: Path, state: dict) -> None:
    """Replace a complete checkpoint only after its bytes reach disk."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(state, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def restore_checkpoint(path: Path, *, work_root: Path, task: dict, resolution: dict | None = None) -> dict:
    """Resume only matching tasks with persistent work and resolved pending effects."""
    path = path.resolve()
    if not path.is_relative_to(work_root.resolve()):
        raise ValueError("Checkpoint outside configured work root")
    state = json.loads(path.read_text())
    if state["task"] != task:
        raise ValueError("Checkpoint task does not match frozen request")
    for name in ("public", "prediction", "workspace", "reference", "grading"):
        resource = Path(state["paths"][name]).resolve()
        if not resource.is_relative_to(work_root.resolve()) or not resource.is_dir():
            raise ValueError(f"Missing or unsafe persistent path: {name}")
    pending = state.get("pending")
    if pending:
        if pending["kind"] != "model":
            raise ValueError("Unresolved tool effect; never replay automatically")
        if not resolution or resolution != {
            "action_id": pending["action_id"],
            "outcome": "not_forwarded",
            "checkpoint": str(path),
        }:
            raise ValueError("Unresolved model request requires explicit receipt review")
        state["pending"] = None
    return state


def remaining_budget(state: dict, *, timeout_s: float, max_steps: int) -> tuple[float, int]:
    """Deduct prior elapsed time and completed model calls, including interruptions."""
    seconds = timeout_s - state["elapsed_seconds"]
    steps = max_steps - state["next_step"]
    if seconds <= 0 or (steps <= 0 and not state.get("pending_calls") and not state.get("solver_done")):
        raise ValueError("Original task budget exhausted; resume cannot extend it")
    return seconds, steps
