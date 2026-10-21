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
