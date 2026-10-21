# SPDX-License-Identifier: Apache-2.0
"""Explicit derived exclusion for model-free CORE workflow controls; never rescore."""

import copy
import hashlib
import json
from pathlib import Path

from nemo_gym.result_disposition import summarize_dispositions
from responses_api_agents.harbor_agent.measurement_review import apply_measurement_review


HOOK = "benchmarks.corebench.operational_review:review"
REASON = "Model-free Oracle workflow control with an empty answer; not a model-performance measurement."


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def review(*, trial_dir: Path, instance_id: str, trial_result: dict) -> dict:
    declaration_path = trial_dir / "operational-control.json"
    declaration = json.loads(declaration_path.read_text())
    raw = (trial_dir / "raw-result.jsonl").read_bytes()
    record = json.loads(raw)
    report = (trial_dir / "empty-report.json").read_bytes()
    if not (
        declaration["version"] == "core-workflow-control-v1"
        and instance_id.startswith("corebench::capsule-")
        and declaration["instance_id"] == instance_id == record["instance_id"]
        and declaration["raw_file_sha256"] == hashlib.sha256(raw).hexdigest()
        and declaration["trial_sha256"] == digest(trial_result) == digest(record["metadata"])
        and declaration["report_sha256"] == hashlib.sha256(report).hexdigest()
        and json.loads(report) == {}
        and trial_result["config"]["agent"]["name"] == "oracle"
        and trial_result["config"]["agent"].get("import_path") is None
        and trial_result["agent_info"]["name"] == "oracle"
        and record["ng_model_call_capture"]["metrics"]["num_calls"] == 0
        and record["ng_model_call_capture"]["calls"] == []
        and record["response"]["output"] == []
    ):
        raise ValueError("Operational-control evidence does not match exact model-free native result")
    native = trial_result["verifier_result"]["rewards"]
    if native != {"reward": 0.0} or record["reward"] != 0.0:
        raise ValueError("Expected unmodified empty-answer control score")
    return {
        "version": "core-workflow-control-v1",
        "metrics": {"reward": {"status": "invalid", "reason": REASON}},
        "evidence": {
            "purpose": "operational_workflow_control",
            "model_performance_eligible": False,
            "raw_file_sha256": declaration["raw_file_sha256"],
            "trial_sha256": declaration["trial_sha256"],
            "report_sha256": declaration["report_sha256"],
            "declaration_sha256": hashlib.sha256(declaration_path.read_bytes()).hexdigest(),
        },
    }


def write_overlay(raw_path: Path, report_path: Path, output_dir: Path) -> dict:
    """Publish a separate reviewed input/report; original bytes remain authoritative."""
    raw = raw_path.read_bytes()
    record = json.loads(raw)  # Exactly one native record, not an implicitly pooled campaign.
    report = report_path.read_bytes()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "raw-result.jsonl").write_bytes(raw)
    (output_dir / "empty-report.json").write_bytes(report)
    declaration = {
        "version": "core-workflow-control-v1",
        "instance_id": record["instance_id"],
        "raw_file_sha256": hashlib.sha256(raw).hexdigest(),
        "trial_sha256": digest(record["metadata"]),
        "report_sha256": hashlib.sha256(report).hexdigest(),
    }
    (output_dir / "operational-control.json").write_text(json.dumps(declaration, indent=2))
    revised, kind, reason = apply_measurement_review(
        record["metadata"], trial_dir=output_dir, instance_id=record["instance_id"], import_path=HOOK
    )
    if kind != "harbor:scientific_measurement_invalid":
        raise ValueError(f"Operational review refused: {kind}: {reason}")
    overlay = copy.deepcopy(record)
    overlay.update(metadata=revised, mask_sample=True, failure_kind=kind, failure_reason=reason)
    (output_dir / "reviewed-results.jsonl").write_text(json.dumps(overlay) + "\n")
    summary = summarize_dispositions([overlay], expected_count=1)
    (output_dir / "disposition-summary.json").write_text(json.dumps(summary, indent=2))
    assert raw_path.read_bytes() == raw
    return summary
