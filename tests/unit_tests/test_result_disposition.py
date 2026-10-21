# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import json
from pathlib import Path

from nemo_gym.result_disposition import summarize_dispositions
from nemo_gym.reward_profile import compute_aggregate_metrics


def test_zero_failure_unknown_and_review_are_distinct():
    zero = {"reward": 0.0, "mask_sample": False, "_ng_task_index": 0, "_ng_rollout_index": 0}
    invalid = {
        "reward": 1.0,
        "mask_sample": True,
        "_ng_task_index": 1,
        "_ng_rollout_index": 0,
        "metadata": {
            "verifier_result": {"rewards": {"quality": 1.0}},
            "measurement_review": {"version": "v1", "metrics": {"quality": {"status": "invalid", "reason": "weak"}}},
        },
    }
    failure = {
        "reward": 0.0,
        "mask_sample": True,
        "failure_kind": "verifier_error",
        "_ng_task_index": 2,
        "_ng_rollout_index": 0,
    }
    raw = [zero, invalid, failure]
    before = copy.deepcopy(raw)
    report = summarize_dispositions(raw, expected_count=4)
    assert report["counts"]["raw_scored_records"] == 2
    assert report["counts"]["quality_selected_records"] == 1
    assert report["per_metric"]["quality"]["invalid"] == 1
    assert report["attempted_total"] is None
    assert report["provider_dispatch_unresolved_total"] is None
    assert report["scientifically_accepted_total"] is None
    assert raw == before


def test_masked_native_aggregation_keeps_dispositions():
    row = {
        "reward": 1,
        "mask_sample": True,
        "metadata": {
            "verifier_result": {"rewards": {"a": 1}},
            "measurement_review": {"metrics": {"a": {"status": "unresolved", "reason": "new protocol"}}},
        },
    }
    result = compute_aggregate_metrics([row])
    assert result.disposition_summary["per_metric"]["a"]["unresolved"] == 1
    assert not any(k.startswith("mean/") for k in result.key_metrics)


def test_unreviewed_metric_is_not_certified():
    report = summarize_dispositions([{"reward": 0, "metadata": {"verifier_result": {"rewards": {"a": 0}}}}])
    assert report["per_metric"]["a"]["unreviewed"] == 1


def test_native_failure_sidecar_does_not_invent_dispatch():
    report = summarize_dispositions([], failures=[{"_ng_failure_class": "agent_request_failed"}], expected_count=1)
    assert report["counts"]["failure_records"] == 1
    assert report["counts"]["raw_scored_records"] == 0
    assert report["provider_dispatch_unresolved_total"] is None


def test_saved_mdarena_record_is_preserved_when_available():
    path = (
        Path(__file__).resolve().parents[4]
        / "outputs/broad-eval-campaign/foundation/mdarena-gym/measurement-review-v1/reviewed-result.jsonl"
    )
    if not path.exists():
        import pytest

        pytest.skip("Campaign saved artifact not shipped with Gym")
    row = json.loads(path.read_text())
    report = summarize_dispositions([row])
    assert report["counts"]["raw_scored_records"] == 1
    assert report["counts"]["quality_selected_records"] == 0
    assert report["per_metric"]["correctness"]["invalid"] == 1
    assert report["records"][0]["metric_dispositions"]["process"]["raw_score"] == 1


def test_native_cli_publishes_disposition_artifact(tmp_path):
    from unittest.mock import patch

    from nemo_gym.cli.eval import reward_profile

    inputs = tmp_path / "inputs.jsonl"
    outputs = tmp_path / "output.jsonl"
    inputs.write_text(json.dumps({"_ng_task_index": 0, "_ng_rollout_index": 0, "agent_ref": {"name": "test"}}) + "\n")
    outputs.write_text(
        json.dumps({"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 0, "mask_sample": True}) + "\n"
    )
    (tmp_path / "output_failures.jsonl").write_text(
        json.dumps({"_ng_task_index": 0, "_ng_rollout_index": 0, "_ng_failure_class": "agent_request_failed"}) + "\n"
    )
    with patch(
        "nemo_gym.cli.eval.get_global_config_dict",
        return_value={"materialized_inputs_jsonl_fpath": str(inputs), "rollouts_jsonl_fpath": str(outputs)},
    ):
        reward_profile()
    report = json.loads((tmp_path / "output.dispositions.json").read_text())
    assert report["expected_rollout_count"] == 1
    assert report["counts"]["failure_records"] == 1
    assert report["counts"]["raw_scored_records"] == 0
    assert report["records"][1]["failure_class"] == "agent_request_failed"


def test_optional_identity_and_attempt_fields_do_not_inflate_lower_bound():
    report = summarize_dispositions(
        [
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 2,
                "_ng_attempt_index": 1,
                "instance_id": "native-id",
                "agent_ref": {"name": "a"},
                "reward": 0,
            }
        ],
        failures=[{"_ng_task_index": 1, "_ng_rollout_index": 2, "_ng_failure_class": "agent_request_failed"}],
    )
    assert report["observed_logical_rollout_lower_bound"] == 1
    assert len(report["records"]) == 2
    assert report["attempted_total"] is None


def test_empty_stream_has_stable_zero_count_schema():
    summary = summarize_dispositions([])
    assert summary["counts"] == {
        "completed_result_records": 0,
        "failure_records": 0,
        "raw_scored_records": 0,
        "quality_selected_records": 0,
        "masked_result_records": 0,
        "explicitly_not_dispatched_records": 0,
    }
    assert summary["attempted_total"] is None
    assert summary["provider_dispatch_unresolved_total"] is None


def _reviewed_mixed():
    return {
        "reward": 0.8,
        "mask_sample": True,
        "metadata": {
            "verifier_result": {"rewards": {"outcome": 0.8, "process": 1.0}},
            "measurement_review": {
                "version": "review-v1",
                "implementation": {"import_path": "bench.review:review", "sha256": "a" * 64},
                "metrics": {
                    "outcome": {"status": "valid", "reason": "native deterministic result inspected"},
                    "process": {"status": "unresolved", "reason": "missing observed evidence"},
                },
            },
        },
    }


def test_per_metric_review_preserves_valid_outcome_despite_whole_row_mask():
    row = _reviewed_mixed()
    before = copy.deepcopy(row)
    aggregate = compute_aggregate_metrics([row])
    metrics = aggregate.disposition_summary["per_metric"]
    assert metrics["outcome"]["reviewed_mean"] == 0.8
    assert metrics["outcome"]["reviewed_denominator"] == 1
    assert metrics["process"]["reviewed_mean"] is None
    assert metrics["process"]["unresolved"] == 1
    assert not any(k.startswith("mean/") for k in aggregate.key_metrics)
    assert row == before


def test_unidentified_review_cannot_produce_reviewed_mean():
    row = _reviewed_mixed()
    del row["metadata"]["measurement_review"]["implementation"]
    metric = summarize_dispositions([row])["per_metric"]["outcome"]
    assert metric["valid"] == 1
    assert metric["valid_but_ineligible_scores"] == 1
    assert metric["reviewed_mean"] is None


def test_review_cannot_certify_nonfinite_or_missing_numeric_source():
    for value in [None, True, float("inf"), float("nan"), "0.8"]:
        row = _reviewed_mixed()
        row["metadata"]["verifier_result"]["rewards"]["outcome"] = value
        metric = summarize_dispositions([row])["per_metric"]["outcome"]
        assert metric["reviewed_denominator"] == 0
        assert metric["reviewed_mean"] is None


def test_distinct_review_sources_are_not_pooled():
    first, second = _reviewed_mixed(), _reviewed_mixed()
    second["metadata"]["measurement_review"]["implementation"]["sha256"] = "b" * 64
    second["metadata"]["verifier_result"]["rewards"]["outcome"] = 0
    metric = summarize_dispositions([first, second])["per_metric"]["outcome"]
    assert metric["reviewed_mean"] is None
    assert metric["reviewed_denominator"] == 2
    assert [g["mean"] for g in metric["reviewed_source_groups"]] == [0.8, 0]


def test_failure_stream_valid_metadata_is_not_completed_reviewed_score():
    metric = summarize_dispositions([], failures=[_reviewed_mixed()])["per_metric"]["outcome"]
    assert metric["reviewed_mean"] is None
    assert metric["reviewed_denominator"] == 0


def test_same_review_producer_cannot_pool_different_datasets_or_agents():
    first, second = _reviewed_mixed(), _reviewed_mixed()
    first["agent_ref"] = {"name": "a"}
    second["agent_ref"] = {"name": "b"}
    first["metadata"]["source"] = "dataset-one"
    second["metadata"]["source"] = "dataset-two"
    metric = summarize_dispositions([first, second])["per_metric"]["outcome"]
    assert metric["reviewed_mean"] is None
    assert len(metric["reviewed_source_groups"]) == 2


def test_missing_dataset_scope_does_not_pool_across_records():
    metric = summarize_dispositions([_reviewed_mixed(), _reviewed_mixed()])["per_metric"]["outcome"]
    assert metric["reviewed_mean"] is None
    assert len(metric["reviewed_source_groups"]) == 2


def test_shared_agent_or_empty_evidence_does_not_establish_dataset_scope():
    first, second = _reviewed_mixed(), _reviewed_mixed()
    for row in (first, second):
        row["agent_ref"] = {"name": "shared"}
        row["metadata"]["measurement_review"]["evidence"] = {}
    metric = summarize_dispositions([first, second])["per_metric"]["outcome"]
    assert metric["reviewed_mean"] is None
    assert len(metric["reviewed_source_groups"]) == 2
