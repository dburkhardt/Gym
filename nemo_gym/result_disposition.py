# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evidence-only accounting alongside native quality metrics; never rescore records."""

import hashlib
import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence


def _numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def summarize_dispositions(
    results: Sequence[Mapping], *, failures: Sequence[Mapping] = (), expected_count: int | None = None
) -> dict:
    """Describe observed records without equating missing output with no dispatch.

    A result is completed collection output, not proof of successful scientific
    execution. Only explicit measurement reviews certify individual metrics.
    Attempt and provider-call totals require their own ledgers and remain unknown.
    """
    counts = Counter(
        {
            name: 0
            for name in (
                "completed_result_records",
                "failure_records",
                "raw_scored_records",
                "quality_selected_records",
                "masked_result_records",
                "explicitly_not_dispatched_records",
            )
        }
    )
    metrics = {}
    reviewed_values = {}
    lineage = []
    observed_identities = set()
    for stream, records in (("results", results), ("failures", failures)):
        for index, record in enumerate(records):
            identity = {
                k: record[k]
                for k in ("_ng_task_index", "_ng_rollout_index", "_ng_attempt_index", "agent_ref", "instance_id")
                if k in record
            }
            if "_ng_task_index" in identity and "_ng_rollout_index" in identity:
                observed_identities.add((identity["_ng_task_index"], identity["_ng_rollout_index"]))
            metadata = record.get("metadata") or {}
            metadata = metadata if isinstance(metadata, dict) else {}
            verifier = metadata.get("verifier_result") or {}
            verifier = verifier if isinstance(verifier, dict) else {}
            native = verifier.get("rewards") or {}
            native = native if isinstance(native, dict) else {}
            review = metadata.get("measurement_review") or {}
            review = review if isinstance(review, dict) else {}
            dispositions = review.get("metrics") or {}
            dispositions = dispositions if isinstance(dispositions, dict) else {}
            selected = stream == "results" and not record.get("mask_sample", False)
            raw_scored = any(_numeric(v) for v in native.values()) or (
                selected and not record.get("_ng_failure_class") and _numeric(record.get("reward"))
            )
            counts["completed_result_records"] += stream == "results"
            counts["failure_records"] += stream == "failures"
            counts["raw_scored_records"] += raw_scored
            counts["quality_selected_records"] += selected
            counts["masked_result_records"] += stream == "results" and bool(record.get("mask_sample"))
            counts["explicitly_not_dispatched_records"] += record.get("_ng_dispatch_drained") is True
            row_metrics = {}
            for name in sorted(set(native) | set(dispositions)):
                disposition = dispositions.get(name) or {}
                disposition = disposition if isinstance(disposition, dict) else {}
                status = disposition.get("status")
                if status not in {"valid", "invalid", "unresolved"}:
                    status = "unreviewed"
                bucket = metrics.setdefault(
                    name,
                    Counter(
                        {
                            key: 0
                            for key in (
                                "valid",
                                "invalid",
                                "unresolved",
                                "unreviewed",
                                "raw_numeric_scores",
                                "reviewed_eligible_scores",
                                "valid_but_ineligible_scores",
                            )
                        }
                    ),
                )
                bucket[status] += 1
                bucket["raw_numeric_scores"] += _numeric(native.get(name))
                implementation = review.get("implementation") or {}
                implementation = implementation if isinstance(implementation, dict) else {}
                source_sha = implementation.get("sha256")
                source_path = implementation.get("import_path")
                identified = (
                    isinstance(review.get("version"), str)
                    and bool(review["version"].strip())
                    and isinstance(source_path, str)
                    and ":" in source_path
                    and isinstance(source_sha, str)
                    and re.fullmatch(r"[0-9a-f]{64}", source_sha) is not None
                )
                eligible = (
                    stream == "results"
                    and status == "valid"
                    and identified
                    and isinstance(disposition.get("reason"), str)
                    and bool(disposition["reason"].strip())
                    and _numeric(native.get(name))
                )
                bucket["reviewed_eligible_scores"] += bool(eligible)
                bucket["valid_but_ineligible_scores"] += status == "valid" and not eligible
                if eligible:
                    # A review producer can serve unrelated benchmarks. Never pool
                    # across declared datasets/agents or native scorer/config variants.
                    scope = {
                        "agent_ref": record.get("agent_ref"),
                        "dataset": metadata.get("source"),
                        "instance_id": record.get("instance_id"),
                        "native_config_sha256": (
                            hashlib.sha256(
                                json.dumps(metadata["config"], sort_keys=True, separators=(",", ":")).encode()
                            ).hexdigest()
                            if metadata.get("config") is not None
                            else None
                        ),
                        "review_evidence": review.get("evidence"),
                    }
                    if not (
                        isinstance(scope["dataset"], str)
                        and bool(scope["dataset"].strip())
                        and scope["native_config_sha256"] is not None
                    ):
                        scope["unscoped_record"] = {"stream": stream, "record_index": index}
                    scope_json = json.dumps(scope, sort_keys=True, separators=(",", ":"))
                    source = (review["version"], source_path, source_sha, scope_json)
                    reviewed_values.setdefault(name, {}).setdefault(source, []).append(native[name])
                row_metrics[name] = {
                    "status": status,
                    "raw_score": native.get(name),
                    "reason": disposition.get("reason"),
                    "reviewed_mean_eligible": bool(eligible),
                }
            lineage.append(
                {
                    "stream": stream,
                    "record_index": index,
                    "identity": identity,
                    "record_sha256": hashlib.sha256(
                        json.dumps(dict(record), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
                    ).hexdigest(),
                    "quality_selected": selected,
                    "raw_scored": raw_scored,
                    "failure_kind": record.get("failure_kind"),
                    "failure_class": record.get("_ng_failure_class"),
                    "review_version": review.get("version"),
                    "review_implementation": review.get("implementation"),
                    "metric_dispositions": row_metrics,
                }
            )
    per_metric = {}
    for name, count in metrics.items():
        groups = reviewed_values.get(name, {})
        group_means = [
            {
                "review_version": key[0],
                "implementation": {"import_path": key[1], "sha256": key[2]},
                "scope": json.loads(key[3]),
                "denominator": len(values),
                "mean": statistics.fmean(values),
            }
            for key, values in sorted(groups.items())
        ]
        per_metric[name] = {
            **dict(count),
            "reviewed_denominator": sum(len(v) for v in groups.values()),
            "reviewed_mean": group_means[0]["mean"] if len(group_means) == 1 else None,
            "reviewed_source_groups": group_means,
        }
    return {
        "schema_version": "2",
        "expected_rollout_count": expected_count,
        "counts": dict(counts),
        "observed_logical_rollout_lower_bound": len(observed_identities),
        "attempted_total": None,
        "provider_dispatch_unresolved_total": None,
        "scientifically_accepted_total": None,
        "limitations": [
            "Missing result rows do not establish whether dispatch occurred.",
            "Logical rollout lower bound merges attempts and optional agent fields within this input run; it is not an attempt total.",
            "Quality selection follows native mask policy; absent metric review is not scientific acceptance.",
            "Provider dispatch state requires its durable ledger; no text-based inference is made.",
            "Reviewed means require an explicit valid metric and identified review producer; this is not independent scientific certification.",
            "Different producer/dataset/agent/native-config/evidence scopes are not pooled; absent scope is record-local.",
        ],
        "per_metric": per_metric,
        "records": lineage,
    }
