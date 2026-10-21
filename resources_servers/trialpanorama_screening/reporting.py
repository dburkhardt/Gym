# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate creator batch means, population weighting, and qualified strict metrics."""

import collections
import hashlib
import json
from pathlib import Path

from resources_servers.trialpanorama_screening.grading import grade


def signature(row):
    return json.dumps(
        {
            "messages": [(x["role"], x["content"]) for x in row["responses_create_params"]["input"]],
            "metadata": row["verifier_metadata"],
        },
        sort_keys=True,
    )


def aggregate_saved(rows, *, frozen_rows, creator_root):
    if not rows or collections.Counter(map(signature, rows)) != collections.Counter(map(signature, frozen_rows)):
        raise ValueError("Saved rows do not exactly cover frozen input/reference population")
    ids = [row["verifier_metadata"]["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task identity")
    items = []
    for row in rows:
        response = row.get("response")
        if not isinstance(response, dict):
            raise ValueError("Missing saved response; cannot reconstruct creator aggregate")
        if response.get("status") != "completed":
            raise ValueError("Incomplete provider response requires separate outcome review")
        text = "\n".join(c.get("text", "") for o in response.get("output", []) for c in o.get("content", []))
        result = grade(text, creator_root=creator_root, **row["verifier_metadata"])
        raw = result["diagnostics"]["creator"]
        if raw != row["diagnostics"]["creator"] or result["reward"] != row["reward"]:
            raise ValueError("Saved creator metric differs from exact source replay")
        valid = result["diagnostics"]["strict_usable"]
        unresolved = result["metadata"]["measurement_review"]["evidence"]["duplicate_reference_pmid"]
        decisions = [x["is_correct"] for x in raw["evaluations"] if x["is_correct"] is not None]
        items.append(
            {
                "task_id": row["verifier_metadata"]["id"],
                "historical_outcome": {
                    key: row[key]
                    for key in (
                        "mask_sample",
                        "failure_kind",
                        "failure_reason",
                        "outcome",
                        "outcome_metadata",
                        "disposition",
                        "model_outcome",
                    )
                    if key in row
                },
                "creator_micro": raw["micro_accuracy"],
                "creator_macro": raw["macro_accuracy"],
                "correct_decisions": sum(decisions),
                "scored_decisions": len(decisions),
                "source_label_slots": len(row["verifier_metadata"]["sample"]["labels"]),
                "strict_usable": valid,
                "reference_unresolved": unresolved,
                "strict_micro": None if unresolved else raw["micro_accuracy"] if valid else 0.0,
                "strict_macro": None if unresolved else raw["macro_accuracy"] if valid else 0.0,
            }
        )
    count = len(items)
    eligible = [r for r in items if not r["reference_unresolved"]]
    scored = sum(r["scored_decisions"] for r in items)
    correct = sum(r["correct_decisions"] for r in items)
    source = Path(creator_root) / "benchmark/tasks/study_screening.py"
    return {
        "version": "trialpanorama-saved-aggregate-v1",
        "creator_commit": "20e97cbb823003fccc425732d593cf2bb205d163",
        "creator_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reporting_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "attempted_batches": count,
        "creator_raw": {
            "avg_micro_accuracy": sum(r["creator_micro"] for r in items) / count,
            "avg_macro_accuracy": sum(r["creator_macro"] for r in items) / count,
            "denominator_batches": count,
            "qualification": "Raw creator means retain fallback behavior and unresolved reference batches",
        },
        "population_weighted": {
            "accuracy": correct / scored if scored else None,
            "correct_decisions": correct,
            "scored_decisions": scored,
            "source_label_slots": sum(r["source_label_slots"] for r in items),
            "qualification": "Distinct metric; prediction-weighted, not creator mean batch micro and not necessarily all source labels",
        },
        "complete_output_v1": {
            "avg_micro_accuracy": sum(r["strict_micro"] for r in eligible) / len(eligible) if eligible else None,
            "avg_macro_accuracy": sum(r["strict_macro"] for r in eligible) / len(eligible) if eligible else None,
            "eligible_batches": len(eligible),
            "model_invalid_batches_scored_zero": sum(not r["strict_usable"] for r in eligible),
            "unresolved_task_ids": [r["task_id"] for r in items if r["reference_unresolved"]],
            "qualification": "Separate explicit variant; malformed/incomplete model output counts zero, source-duplicate references unresolved and excluded only from this qualified aggregate",
        },
        "items": items,
    }
