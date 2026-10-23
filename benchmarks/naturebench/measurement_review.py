# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exclude errored evaluator measurements without replacing creator scores."""

import hashlib
import json
import math
from pathlib import Path


def _errors(value: object, path: str = "raw_scores") -> list[str]:
    if isinstance(value, dict):
        errors = [f"{path}.error: {value['error']}"] if value.get("error") else []
        for key, child in value.items():
            if key != "error":
                errors.extend(_errors(child, f"{path}.{key}"))
        return errors
    if isinstance(value, list):
        return [error for index, child in enumerate(value) for error in _errors(child, f"{path}[{index}] ")]
    return []


def _molecular_schema(raw: object, instance_id: str) -> bool:
    if instance_id != "naturebench::s42256-024-00916-5" or not isinstance(raw, dict):
        return False
    required = {
        "drd2_qed": {"SR@0.4", "SR@0.5", "SR@0.6"},
        "admet": {
            "SRA_hERG",
            "SRA_ESOL",
            "SRA_BBBP",
            "SRA_Mutagenicity",
            "SRA_lipop",
            "SRA_hERG ESOL",
            "SRA_Mutagenicity BBBP",
            "SRA_hERG Mutagenicity",
            "SRA_Mutagenicity ESOL",
            "SRA_Mutagenicity lipop",
            "SRA_overall",
        },
    }
    if set(raw) != set(required):
        return False
    for group, metrics in required.items():
        values = raw[group]
        if not isinstance(values, dict) or not metrics.issubset(values):
            return False
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
            for key, value in values.items()
            if key != "error"
        ):
            return False
    return True


def review(*, trial_dir: Path, instance_id: str, trial_result: dict) -> dict:
    """Review the creator-selected best attempt; earlier failed attempts do not invalidate a later valid best."""
    native = (trial_result.get("verifier_result") or {}).get("rewards") or {}
    path = trial_dir / "artifacts/state/best_score.json"
    result = {"version": "naturebench-evaluator-validity-v1", "metrics": {}, "evidence": {"instance_id": instance_id}}
    if not path.is_file():
        status, reason = "unresolved", "Creator best_score.json is missing; numeric reward is not verified."
    else:
        try:
            best = json.loads(path.read_text())
        except (ValueError, UnicodeError):
            best = None
        if not isinstance(best, dict):
            result["metrics"] = {
                name: {"status": "unresolved", "reason": "Malformed creator best-score object."} for name in native
            }
            return result
        result["evidence"].update(
            best_score_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), best_attempt=best.get("best_attempt")
        )
        errors = _errors(best.get("best_raw_scores"))
        if errors:
            status, reason = (
                "invalid",
                "Evaluator infrastructure/scoring errors in selected best attempt: " + "; ".join(errors),
            )
        elif best.get("finalized") is not True or not best.get("best_raw_scores"):
            status, reason = "unresolved", "Creator evaluator has no finalized nonempty raw measurement."
        elif not _molecular_schema(best.get("best_raw_scores"), instance_id):
            status, reason = (
                "unresolved",
                "Unsupported or incomplete molecular raw-score schema; required metrics must be finite numbers.",
            )
        else:
            status, reason = (
                "valid",
                "Selected finalized creator evaluation contains no reported evaluator errors; raw reward retained.",
            )
    result["metrics"] = {name: {"status": status, "reason": reason} for name in native}
    return result
