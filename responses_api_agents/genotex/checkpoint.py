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
        if resolution and resolution.get("outcome") == "settled_error":
            record = settled_error_resolution(path, state, resolution, work_root)
            state.setdefault("settled_model_errors", []).append(record)
            state["pending"] = None
            return state
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


def settled_error_resolution(path: Path, state: dict, resolution: dict, work_root: Path) -> dict:
    """Validate an explicit retained gateway error packet; never infer no forwarding.

    This validates provenance consistency, not authenticity of arbitrary supplied
    files. The caller selects a reviewed local packet. Runtime/config equality is
    still enforced by the agent before any new dispatch.
    """
    import hashlib
    from datetime import datetime

    if set(resolution) != {"outcome", "checkpoint", "action_id", "evidence", "evidence_sha256"}:
        raise ValueError("Malformed settled-error resolution")
    pending = state.get("pending") or {}
    if resolution["outcome"] != "settled_error" or resolution["checkpoint"] != str(path):
        raise ValueError("Settled-error checkpoint mismatch")
    if pending.get("kind") != "model" or resolution["action_id"] != pending.get("action_id"):
        raise ValueError("Settled-error action mismatch")
    if state.get("pending_calls") or pending.get("step") != state.get("next_step"):
        raise ValueError("Settled-error state is not a model boundary")
    if not state.get("runtime_identity") or not state.get("guard_config_sha256"):
        raise ValueError("Settled-error recovery requires recorded runtime/config identity")

    def load(filename, digest):
        target = Path(filename).resolve()
        if not target.is_relative_to(work_root.resolve()) or not target.is_file():
            raise ValueError("Unsafe settled-error evidence path")
        raw = target.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Settled-error evidence hash mismatch")
        return json.loads(raw)

    evidence = load(resolution["evidence"], resolution["evidence_sha256"])
    if evidence.get("version") != 1 or evidence.get("wire_encoding") != "python-json-dumps-default-utf8-v1":
        raise ValueError("Unsupported settled-error packet/serialization contract")
    if evidence["checkpoint_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Settled-error checkpoint hash mismatch")
    if evidence["action_id"] != pending["action_id"]:
        raise ValueError("Settled-error packet action mismatch")
    artifacts = {name: load(**evidence[name]) for name in ("request", "response", "started", "receipt", "gateway")}
    request, response, started, receipt, gateway = (
        artifacts[n] for n in ("request", "response", "started", "receipt", "gateway")
    )
    if request.get("input") != state["history"]:
        raise ValueError("Settled-error request history mismatch")
    wire_hash = hashlib.sha256(json.dumps(request).encode()).hexdigest()
    starts = [r for r in gateway if r.get("event") == "started" and r.get("request_sha256") == wire_hash]
    if len(starts) != 1:
        raise ValueError("Settled-error requires a unique wire request")
    start = starts[0]
    terminal = [
        r
        for r in gateway
        if r.get("request_id") == start["request_id"] and r.get("event") in ("finished", "transport_error")
    ]
    if len(terminal) != 1 or terminal[0].get("event") != "finished":
        raise ValueError("Settled-error requires one terminal HTTP response")
    end = terminal[0]
    status = end.get("http_status")
    if (
        type(status) is not int
        or status not in (429, 500, 502, 503, 504)
        or type(receipt.get("status")) is not int
        or receipt.get("status") != status
    ):
        raise ValueError("Settled-error HTTP status mismatch")
    if not isinstance(response, dict) or not isinstance(response.get("error"), dict) or response.get("output"):
        raise ValueError("Settled-error packet must retain an error response")
    if str(response["error"].get("code")) != str(status):
        raise ValueError("Settled-error response code mismatch")
    request_parent = Path(evidence["request"]["filename"]).resolve().parent
    if any(Path(evidence[n]["filename"]).resolve().parent != request_parent for n in ("response", "started")):
        raise ValueError("Settled-error local artifact lineage mismatch")
    if receipt.get("request_artifact_id") != request_parent.name:
        raise ValueError("Settled-error local receipt identity mismatch")
    if not (
        datetime.fromisoformat(started["utc"]).timestamp()
        <= start["time"]
        <= end["time"]
        <= datetime.fromisoformat(receipt["utc"]).timestamp()
    ):
        raise ValueError("Settled-error receipt time mismatch")
    if request.get("model") != start.get("model") or request.get("model") != receipt.get("requested_model"):
        raise ValueError("Settled-error model mismatch")
    return {
        "action_id": pending["action_id"],
        "step": pending["step"],
        "outcome": "settled_error",
        "http_status": status,
        "gateway_request_id": start["request_id"],
        "wire_sha256": wire_hash,
        "evidence_sha256": resolution["evidence_sha256"],
        "request": request,
    }
