# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned review of audited frozen task measurements; raw RewardKit scores stay intact."""

import hashlib
import json
from pathlib import Path


AUDITED_INPUT_SHA256 = "32965463e77ad6106ae92759f2701b558a1408e3b234c0d42f8c7a4c6d45eef9"
AUDITED_CHECK_SHA256 = "47383a53ea8f4eb6bcd8e2bea11881229169073e7241dd39792a83059139529a"
AUDITED_CONFIG_SHA256 = {
    "1e15658ef0379a3bb2b6c51d07146eaa1538ec9509ec42d85661532ebc76a592",
    "d59eca56ac26355646a72d79faedb7d809571bf468352131d5d7596e392ade05",
}


TASK18_CHECK_SHA256 = "4742b137ec647c4e79e5a2c62ce0f941616ef4728583852099d397167ed20df6"
TASK18_CRITERIA = {
    "ffnonbonded_ion_atom_type_charges_are_scaled",
    "ions_itp_molecule_charges_are_scaled",
    "unused_ion_parameters_are_unchanged",
    "scaled_topology_points_at_the_scaled_force_field",
    "unscaled_force_field_files_are_unchanged",
}


def _review_task18(trial_dir: Path, trial_result: dict) -> dict:
    result = {"version": "mdarena-task18-preservation-review-v1", "metrics": {}, "evidence": {}}
    native = (trial_result.get("verifier_result") or {}).get("rewards") or {}
    if "correctness" not in native:
        return result
    controller = trial_dir / "verifier/native-controller/controller"
    values = {}
    for key, path in {
        "identity": controller.parent / "identity.json",
        "details": controller / "result/reward-details.json",
    }.items():
        if path.exists():
            payload = path.read_bytes()
            result["evidence"][str(path.relative_to(trial_dir))] = hashlib.sha256(payload).hexdigest()
            values[key] = json.loads(payload)
    identity = values.get("identity", {})
    criteria = values.get("details", {}).get("correctness", {}).get("criteria", [])
    names = {item.get("name") for item in criteria if item.get("weight", 1) > 0}
    known = (
        identity.get("test_sources_sha256", {}).get("correctness/checks.py") == TASK18_CHECK_SHA256
        and names == TASK18_CRITERIA
    )
    result["metrics"]["correctness"] = {
        "status": "unresolved",
        "reason": (
            "Audited task18 checks detect tested charge, unused-ion, include and bonded-parameter errors, "
            "but accept doubled sodium Lennard-Jones sigma and changed sodium molecule count. "
            "Raw correctness remains a score for those checks, not certification of full parameter/composition "
            "preservation. Process judging was not exercised by these controls."
            if known
            else "Task18 scorer source or criterion signature is changed or unavailable; preservation qualification "
            "requires new source-bound review. No process-judge conclusion is implied."
        ),
    }
    return result


def review(*, trial_dir: Path, instance_id: str, trial_result: dict) -> dict:
    """Do not extrapolate one audited task's scientific findings to other tasks."""
    if instance_id == "mdarena::18_scale_charge":
        return _review_task18(trial_dir, trial_result)
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
