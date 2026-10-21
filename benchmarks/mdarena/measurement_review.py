# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned review of frozen task02 measurements; raw RewardKit scores stay intact."""

import hashlib
import json
from pathlib import Path


AUDITED_INPUT_SHA256 = "32965463e77ad6106ae92759f2701b558a1408e3b234c0d42f8c7a4c6d45eef9"
AUDITED_CHECK_SHA256 = "47383a53ea8f4eb6bcd8e2bea11881229169073e7241dd39792a83059139529a"
AUDITED_CONFIG_SHA256 = {
    "1e15658ef0379a3bb2b6c51d07146eaa1538ec9509ec42d85661532ebc76a592",
    "d59eca56ac26355646a72d79faedb7d809571bf468352131d5d7596e392ade05",
}


def review(*, trial_dir: Path, instance_id: str, trial_result: dict) -> dict:
    """Do not extrapolate one audited task's scientific findings to other tasks."""
    result = {"version": "mdarena-task02-scientific-review-v1", "metrics": {}, "evidence": {}}
    if instance_id != "mdarena::02_conc_diff_sys_traj":
        return result
    controller = trial_dir / "verifier/native-controller/controller"
    details_path = controller / "result/reward-details.json"
    details = json.loads(details_path.read_text())
    result["evidence"][str(details_path.relative_to(trial_dir))] = hashlib.sha256(
        details_path.read_bytes()
    ).hexdigest()
    identity_path = controller.parent / "identity.json"
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    if identity_path.exists():
        result["evidence"][str(identity_path.relative_to(trial_dir))] = hashlib.sha256(
            identity_path.read_bytes()
        ).hexdigest()
    known_source = (
        identity.get("configuration_sha256") in AUDITED_CONFIG_SHA256
        or identity.get("test_sources_sha256", {}).get("correctness/checks.py") == AUDITED_CHECK_SHA256
    )
    native = (trial_result.get("verifier_result") or {}).get("rewards") or {}
    if "correctness" in native:
        names = {x.get("name") for x in details.get("correctness", {}).get("criteria", [])}
        known = known_source and names == {"declined_to_concatenate_and_reported_why"}
        result["metrics"]["correctness"] = {
            "status": "invalid" if known else "unresolved",
            "reason": "Audited task02 criterion accepts irrelevant nonempty final answers and absence of concatenated trajectory; this raw score does not establish scientific correctness."
            if known
            else "Task02 correctness criterion changed or is unavailable; new scientific validation is required.",
        }
    if "process" in native:
        requests = sorted((controller / "judge-receipts").glob("*/*-request.json"))
        observations = 0
        audited_inputs = []
        for path in requests:
            body = json.loads(path.read_text())
            result["evidence"][str(path.relative_to(trial_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
            audited_inputs.append(
                hashlib.sha256(json.dumps(body.get("input"), sort_keys=True).encode()).hexdigest()
                == AUDITED_INPUT_SHA256
            )
            observations += sum(
                item.get("type") == "function_call_output" and bool(item.get("output"))
                for item in body.get("input", [])
                if isinstance(item, dict)
            )
        result["metrics"]["process"] = {
            "status": "invalid"
            if len(requests) == 1 and audited_inputs == [True] and not observations
            else "unresolved",
            "reason": "Audited task02 judge has no recorded filesystem/tool observations; its rubric-and-path judgment is unsupported by inspected evidence."
            if len(requests) == 1 and audited_inputs == [True] and not observations
            else "Task02 process judgment needs independent evidence-grounding validation; tool observations alone do not establish correctness.",
            "recorded_tool_observations": observations,
        }
    return result
