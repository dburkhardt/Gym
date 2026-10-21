# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.result_disposition import summarize_dispositions
from resources_servers.campaign_creator_scoring import app


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "task_id,answer,invalid", [("233", r"\boxed{3.51e-09}", True), ("294", r"\boxed{99999}", False)]
)
async def test_actual_pinned_qcbench_measurement(task_id, answer, invalid, tmp_path):
    base = Path("../../outputs/broad-eval-campaign")
    source = Path("../qcbench-upstream").resolve()
    samples = base / "scientific-agents/qcbench/qcbench-50.jsonl"
    fixture = base / "moljson-schema-repaired/pilot-5-v2.jsonl"
    if not all(p.exists() for p in (source, samples, fixture)):
        pytest.skip("Pinned creator source and campaign samples unavailable")
    sample = next(json.loads(line) for line in samples.read_text().splitlines() if json.loads(line)["uuid"] == task_id)
    saved = next(
        json.loads(line)
        for line in fixture.read_text().splitlines()
        if json.loads(line)["response"]["status"] == "incomplete"
    )
    response = copy.deepcopy(saved["response"])
    response["output"] = [
        {
            "id": "control",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": answer, "annotations": []}],
        }
    ]
    body = app.VerifyRequest.model_validate(
        {
            **sample,
            "response": response,
            "_ng_task_index": 7,
            "_ng_rollout_index": 2,
            "_ng_attempt_index": 1,
            "metadata": {"caller_note": "preserve", "measurement_review": {"stale": True}},
            "mask_sample": True,
            "reward": 99,
        }
    )
    server = SimpleNamespace(
        config=SimpleNamespace(benchmark="qcbench", upstream_repo=str(source), scoring_timeout_s=30)
    )
    result = await app.CreatorResourcesServer.verify(server, body)
    dumped = result.model_dump(mode="json")
    assert [dumped[k] for k in ("_ng_task_index", "_ng_rollout_index", "_ng_attempt_index")] == [7, 2, 1]
    assert result.metadata["caller_note"] == "preserve"
    assert "stale" not in result.metadata["measurement_review"]
    assert result.reward == 0
    assert result.mask_sample is invalid
    assert result.scoring_details["reference_self_score"] == (0 if invalid else 1)
    assert result.response == body.response
    report = summarize_dispositions([result.model_dump(mode="json")])
    assert report["counts"]["quality_selected_records"] == (0 if invalid else 1)
    assert report["per_metric"]["creator_numeric"]["invalid" if invalid else "unresolved"] == 1
    assert report["per_metric"]["creator_numeric"]["reviewed_mean"] is None
    assert result.failure_kind == ("qcbench:invalid_reference_measurement" if invalid else "model_budget")


@pytest.mark.asyncio
async def test_mismatched_task_evidence_fails_closed(monkeypatch):
    from unittest.mock import AsyncMock

    from nemo_gym.openai_utils import NeMoGymResponse

    body = app.VerifyRequest.model_construct(
        uuid="233",
        verifier_metadata={"answer": "3.51e-09"},
        response=NeMoGymResponse.model_construct(output=[], status="completed"),
    )
    monkeypatch.setattr(
        app,
        "run_score",
        AsyncMock(
            return_value={
                "reward": 0,
                "measurement_valid": False,
                "measurement_evidence": {"task_sha256": "wrong"},
            }
        ),
    )
    server = SimpleNamespace(config=SimpleNamespace(benchmark="qcbench", upstream_repo="/tmp", scoring_timeout_s=1))
    with pytest.raises(ValueError, match="matching task/reference evidence"):
        await app.CreatorResourcesServer.verify(server, body)
