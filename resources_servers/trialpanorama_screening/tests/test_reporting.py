# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import json
import os

import pytest

from resources_servers.trialpanorama_screening.grading import grade
from resources_servers.trialpanorama_screening.reporting import aggregate_saved


def row(task, labels, predicted):
    sample = {
        "context": [{"PMID": str(i), "title": "fixture", "abstract": "fixture"} for i in range(len(labels))],
        "labels": [{"PMID": str(i), "label": label} for i, label in enumerate(labels)],
    }
    text = (
        json.dumps({"results": [{"PMID": str(i), "decision": label} for i, label in enumerate(predicted)]})
        if predicted
        else ""
    )
    meta = {"sample": sample, "id": task}
    result = grade(text, creator_root=os.environ["TRIALPANORAMA_SOURCE"], **meta)
    return dict(
        responses_create_params={"input": [{"role": "user", "content": "same fixture"}]},
        verifier_metadata=meta,
        response={"status": "completed", "output": [{"content": [{"text": text}]}]},
        **result,
    )


def test_batch_means_not_population_weighted():
    rows = [row(1, ["included"], ["included"]), row(2, ["included"] * 3, ["excluded"] * 3)]
    result = aggregate_saved(rows, frozen_rows=rows, creator_root=os.environ["TRIALPANORAMA_SOURCE"])
    assert result["creator_raw"]["avg_micro_accuracy"] == 0.5
    assert result["creator_raw"]["avg_macro_accuracy"] == 0.5
    assert result["population_weighted"]["accuracy"] == 0.25


def test_fallback_zero_strict_and_frozen_identity():
    rows = [row(1, ["excluded"], [])]
    result = aggregate_saved(rows, frozen_rows=rows, creator_root=os.environ["TRIALPANORAMA_SOURCE"])
    assert result["creator_raw"]["avg_micro_accuracy"] == 1
    assert result["complete_output_v1"]["avg_micro_accuracy"] == 0
    assert result["complete_output_v1"]["model_invalid_batches_scored_zero"] == 1
    changed = copy.deepcopy(rows)
    changed[0]["responses_create_params"]["input"][0]["content"] = "changed"
    with pytest.raises(ValueError, match="frozen"):
        aggregate_saved(rows, frozen_rows=changed, creator_root=os.environ["TRIALPANORAMA_SOURCE"])


def test_duplicate_reference_stays_raw_but_unresolved_in_strict():
    saved = row(24, ["included"], ["included"])
    saved["verifier_metadata"]["sample"]["labels"] *= 2
    text = saved["response"]["output"][0]["content"][0]["text"]
    saved.update(grade(text, creator_root=os.environ["TRIALPANORAMA_SOURCE"], **saved["verifier_metadata"]))
    result = aggregate_saved([saved], frozen_rows=[saved], creator_root=os.environ["TRIALPANORAMA_SOURCE"])
    assert result["creator_raw"]["avg_micro_accuracy"] == 1
    assert result["creator_raw"]["denominator_batches"] == 1
    assert result["complete_output_v1"]["unresolved_task_ids"] == [24]
    assert result["complete_output_v1"]["eligible_batches"] == 0
    assert result["complete_output_v1"]["avg_micro_accuracy"] is None


def test_historical_outcome_retained_and_missing_response_refused():
    saved = row(1, ["included"], ["included"])
    saved.update(
        mask_sample=True,
        failure_kind="historical:qualification",
        failure_reason="original reason",
        outcome_metadata={"status": "original"},
    )
    report = aggregate_saved([saved], frozen_rows=[saved], creator_root=os.environ["TRIALPANORAMA_SOURCE"])
    assert report["items"][0]["historical_outcome"] == {
        "mask_sample": True,
        "failure_kind": "historical:qualification",
        "failure_reason": "original reason",
        "outcome_metadata": {"status": "original"},
    }
    missing = copy.deepcopy(saved)
    missing.pop("response")
    with pytest.raises(ValueError, match="Missing saved response"):
        aggregate_saved([missing], frozen_rows=[saved], creator_root=os.environ["TRIALPANORAMA_SOURCE"])
