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
    observed_attempts = set()
    observed_tasks = set()
    provider_attempts = {}
    provider_calls = set()
    provider_evidence_records = 0
    provider_evidence_errors = []
    for stream, records in (("results", results), ("failures", failures)):
        for index, record in enumerate(records):
            identity = {
                k: record[k]
                for k in ("_ng_task_index", "_ng_rollout_index", "_ng_attempt_index", "agent_ref", "instance_id")
                if k in record
            }
            if "_ng_task_index" in identity and "_ng_rollout_index" in identity:
                observed_identities.add((identity["_ng_task_index"], identity["_ng_rollout_index"]))
            if "_ng_task_index" in identity:
                observed_tasks.add(identity["_ng_task_index"])
            if (
                "_ng_task_index" in identity
                and "_ng_rollout_index" in identity
                and record.get("_ng_dispatch_drained") is not True
            ):
                observed_attempts.add(
                    (identity["_ng_task_index"], identity["_ng_rollout_index"], identity.get("_ng_attempt_index", 0))
                )
            capture = record.get("ng_model_call_capture")
            if isinstance(capture, dict) and isinstance(capture.get("calls"), list):
                provider_evidence_records += 1
                for call in capture["calls"]:
                    if not isinstance(call, dict) or (
                        not isinstance(call.get("model_call_id"), str) or not call["model_call_id"].strip()
                    ):
                        provider_evidence_errors.append(
                            {"stream": stream, "record_index": index, "reason": "missing model-call identity"}
                        )
                        continue
                    call_id = call["model_call_id"]
                    provider_calls.add(call_id)
                    attempts = call.get("provider_attempts")
                    if not isinstance(attempts, list):
                        provider_evidence_errors.append(
                            {"stream": stream, "record_index": index, "reason": "provider attempts unavailable"}
                        )
                        continue
                    for attempt in attempts:
                        if not isinstance(attempt, dict) or (
                            not isinstance(attempt.get("attempt_id"), str) or not attempt["attempt_id"].strip()
                        ):
                            provider_evidence_errors.append(
                                {
                                    "stream": stream,
                                    "record_index": index,
                                    "reason": "missing provider-attempt identity",
                                }
                            )
                            continue
                        entry = provider_attempts.setdefault(
                            attempt["attempt_id"], {"model_call_ids": set(), "outcomes": set(), "records": set()}
                        )
                        entry["model_call_ids"].add(call_id)
                        entry["outcomes"].add(str(attempt.get("outcome")))
                        entry["records"].add((stream, index))
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
                aggregation = disposition.get("aggregation", "mean")
                supported_aggregation = isinstance(aggregation, str) and aggregation in {"mean", "sqrt_mean"}
                eligible = (
                    stream == "results"
                    and status == "valid"
                    and identified
                    and isinstance(disposition.get("reason"), str)
                    and bool(disposition["reason"].strip())
                    and _numeric(native.get(name))
                    and supported_aggregation
                    and (aggregation != "sqrt_mean" or native[name] >= 0)
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
                    source = (review["version"], source_path, source_sha, scope_json, aggregation)
                    reviewed_values.setdefault(name, {}).setdefault(source, []).append(native[name])
                row_metrics[name] = {
                    "status": status,
                    "raw_score": native.get(name),
                    "reason": disposition.get("reason"),
                    "reviewed_mean_eligible": bool(eligible),
                    "aggregation": aggregation,
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
                    "raw_reward": record.get("reward"),
                    "native_metric_scores": dict(native),
                    "failure_kind": record.get("failure_kind"),
                    "failure_class": record.get("_ng_failure_class"),
                    "review_evidence": review.get("evidence"),
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
                "aggregation": key[4],
                "mean": statistics.fmean(values) if key[4] == "mean" else None,
                "value": (math.sqrt(statistics.fmean(values)) if key[4] == "sqrt_mean" else statistics.fmean(values)),
            }
            for key, values in sorted(groups.items())
        ]
        per_metric[name] = {
            **dict(count),
            "reviewed_denominator": sum(len(v) for v in groups.values()),
            "reviewed_mean": group_means[0]["mean"] if len(group_means) == 1 else None,
            "reviewed_value": group_means[0]["value"] if len(group_means) == 1 else None,
            "reviewed_aggregation": group_means[0]["aggregation"] if len(group_means) == 1 else None,
            "reviewed_source_groups": group_means,
        }
    provider_counts = Counter({"captured": 0, "rejected": 0, "unresolved": 0, "conflicting": 0})
    for evidence in provider_attempts.values():
        outcomes = evidence["outcomes"]
        if len(outcomes) != 1 or len(evidence["model_call_ids"]) != 1:
            provider_counts["conflicting"] += 1
            provider_counts["unresolved"] += 1
        elif outcomes == {"captured"}:
            provider_counts["captured"] += 1
        elif outcomes == {"rejected"}:
            provider_counts["rejected"] += 1
        else:
            provider_counts["unresolved"] += 1
    return {
        "schema_version": "3",
        "expected_rollout_count": expected_count,
        "counts": dict(counts),
        "observed_logical_rollout_lower_bound": len(observed_identities),
        "observed_distinct_task_lower_bound": len(observed_tasks),
        "observed_rollout_attempt_lower_bound": len(observed_attempts),
        "provider_evidence": {
            "scope": "model-call evidence embedded in these exact input result/failure records only",
            "records_with_capture_evidence": provider_evidence_records,
            "logical_model_call_lower_bound": len(provider_calls),
            "physical_attempt_lower_bound": len(provider_attempts),
            "outcome_counts": dict(provider_counts),
            "attempts": [
                {
                    "attempt_id": key,
                    "model_call_ids": sorted(value["model_call_ids"]),
                    "observed_outcomes": sorted(value["outcomes"]),
                    "source_records": [
                        {"stream": stream, "record_index": index} for stream, index in sorted(value["records"])
                    ],
                }
                for key, value in sorted(provider_attempts.items())
            ],
            "errors": provider_evidence_errors,
        },
        "attempted_total": None,
        "provider_dispatch_unresolved_total": None,
        "scientifically_accepted_total": None,
        "limitations": [
            "Missing result rows do not establish whether dispatch occurred.",
            "Logical rollout lower bound merges attempts and optional agent fields within this input run; it is not an attempt total.",
            "Quality selection follows native mask policy; absent metric review is not scientific acceptance.",
            "Embedded provider evidence is a lower bound; pending-only calls absent from these rows remain unknown.",
            "Distinct task and rollout-attempt lower bounds assume one input run; do not pool reused native indexes across runs.",
            "Conflicting provider outcomes are unresolved, never resolved by guessed chronology.",
            "Reviewed means require an explicit valid metric and identified review producer; this is not independent scientific certification.",
            "Different producer/dataset/agent/native-config/evidence scopes are not pooled; absent scope is record-local.",
        ],
        "per_metric": per_metric,
        "records": lineage,
    }
