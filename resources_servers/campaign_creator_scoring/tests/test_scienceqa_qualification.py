# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.scienceqa.qualification import qualify
from benchmarks.scientific_agent_campaign import deterministic
from resources_servers.campaign_creator_scoring import app


BASE = Path("../../outputs/broad-eval-campaign")
REPO = Path("../scienceqa-upstream").resolve()


def sample(task_id):
    if not REPO.exists():
        pytest.skip("Pinned ScienceQA assets unavailable")
    return next(
        json.loads(line)
        for line in (BASE / "scientific-agents/scienceqa/scienceqa-50.jsonl").read_text().splitlines()
        if json.loads(line)["uuid"] == task_id
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit,fallback", [(False, True), (True, False)])
async def test_actual_native_11434_retains_raw_scores_and_excludes_caption_defect(explicit, fallback):
    from benchmarks.scienceqa.qualification import FORMAT_SUFFIX

    row = sample("11434")
    if explicit:
        row["responses_create_params"]["input"][0]["content"] += FORMAT_SUFFIX
    saved = json.loads((BASE / "scienceqa/pilot-5-v2.jsonl").read_text().splitlines()[0])
    response = copy.deepcopy(saved["response"])
    response["output"] = [
        {
            "id": "fixture",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "A" if fallback else "The answer is A.", "annotations": []}],
        }
    ]
    body = app.VerifyRequest.model_validate({**row, "response": response, "_ng_task_index": 4})
    server = SimpleNamespace(
        config=SimpleNamespace(benchmark="scienceqa", upstream_repo=str(REPO), scoring_timeout_s=30)
    )
    result = await app.CreatorResourcesServer.verify(server, body)
    assert result.mask_sample
    assert result.scoring_details["fallback_randomized"] is fallback
    assert result.response == body.response
    assert result.model_dump()["_ng_task_index"] == 4
    review = result.metadata["measurement_review"]
    assert review["metrics"]["strict_accuracy"]["status"] == "invalid"
    assert review["evidence"]["source_record_matches"]
    assert review["evidence"]["scope"] == "not-established-as-life-sciences"
    assert review["evidence"]["protocol_variant"].endswith("explicit-format" if explicit else "zero-shot")
    assert result.metadata["verifier_result"]["rewards"]["strict_accuracy"] == result.scoring_details["strict_score"]


def test_changed_input_not_certified_and_no_scoring_change():
    row = sample("11434")
    details = deterministic.score("scienceqa", REPO, row, "The answer is A.")
    before = copy.deepcopy(details)
    row["verifier_metadata"]["caption"] = "Changed caption"
    review = qualify(REPO, row, details)
    assert review["metrics"]["strict_accuracy"]["status"] == "unresolved"
    assert not review["evidence"]["source_record_matches"]
    assert details == before


def test_nondefective_wrong_answer_remains_model_failure_not_invalid_measurement():
    path = BASE / "scientific-agents/scienceqa/scienceqa-5.jsonl"
    row = json.loads(path.read_text().splitlines()[0])
    wrong = "ABCDE"[(row["verifier_metadata"]["answer"] + 1) % len(row["verifier_metadata"]["choices"])]
    details = deterministic.score("scienceqa", REPO, row, f"The answer is {wrong}.")
    review = qualify(REPO, row, details)
    assert details["reward"] == details["strict_score"] == 0
    assert not details["fallback_randomized"]
    assert {m["status"] for m in review["metrics"].values()} == {"unresolved"}
