# SPDX-License-Identifier: Apache-2.0
"""Keep raw outcome evidence distinct from incomplete process visual coverage."""

import hashlib
import json
import math
from pathlib import Path


def review(*, trial_dir: Path, instance_id: str, trial_result: dict) -> dict:
    record_path = trial_dir / "verifier/biabench/managed-process/score/result.json"
    record = json.loads(record_path.read_text())
    identity_path = record_path.with_name("identity.json")
    identity = json.loads(identity_path.read_text())
    manifest_bytes = Path(__file__).with_name("frozen_sources.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    expected_manifest = hashlib.sha256(manifest_bytes).hexdigest()
    imports = record.get("creator_import_sha256") or {}
    known = (
        identity.get("source_manifest_sha256") == expected_manifest
        and record.get("status") == "completed"
        and instance_id == "biabench::" + manifest["task_id"]
        and bool(imports)
        and all(manifest["creator_files"].get(p) == sha for p, sha in imports.items())
    )
    native = (trial_result.get("verifier_result") or {}).get("rewards") or {}
    scores = record.get("native_scores") or {}

    def matches(name, source_name):
        value = native.get(name)
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value == scores.get(source_name)
        )

    metrics = {}
    for name in ("reward", "outcome"):
        if name in native:
            metrics[name] = {
                "status": "valid" if known and matches(name, "result_score") else "unresolved",
                "reason": "Pinned native outcome scorer completed; this establishes execution fidelity, not universal scientific correctness."
                if known and matches(name, "result_score")
                else "Outcome source/execution identity is unqualified.",
            }
    process = record.get("measurement_validity", {}).get("process", {})
    if "process" in native:
        metrics["process"] = {
            "status": process.get("status", "unresolved")
            if known and matches("process", "checklist_score")
            else "unresolved",
            "reason": process.get("reason", "Process coverage metadata unavailable."),
        }
    else:
        # No numeric process metric exists. Existing hook only accepts names present in rewards;
        # conservatively mark the scalar unresolved while preserving independent raw outcome.
        if "reward" in native:
            metrics["reward"] = {
                "status": "unresolved",
                "reason": "Required process checklist has no numeric score; raw outcome remains independently reported.",
            }
    return {
        "version": "biabench-process-coverage-v1",
        "metrics": metrics,
        "process_evidence": process,
        "aggregation_limit": "Shared review conservatively masks the whole row if any metric is unresolved; raw outcome is retained.",
        "evidence": {
            str(p.relative_to(trial_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (record_path, identity_path)
        },
    }
