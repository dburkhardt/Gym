# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path

import pytest

from resources_servers.evidencebench.app import Request, Verifier
from resources_servers.proteinlmbench.app import ProteinLMBenchRequest, ProteinLMBenchVerifier


@pytest.mark.asyncio
async def test_actual_saved_budget_and_lineage():
    source = Path(os.environ.get("PROTEIN_SAVED_RESULTS", "missing"))
    if not source.exists():
        pytest.skip("Actual saved ProteinLMBench result required")
    raw = next(
        json.loads(line)
        for line in source.read_text().splitlines()
        if json.loads(line)["response"]["status"] == "incomplete"
    )
    result = (await ProteinLMBenchVerifier().verify(ProteinLMBenchRequest.model_validate(raw))).model_dump()
    assert result["reward"] == raw["reward"] and result["mask_sample"] == raw["mask_sample"]
    assert result["failure_kind"] == "model_budget"
    assert (
        result["_ng_task_index"] == raw["_ng_task_index"] and result["_ng_rollout_index"] == raw["_ng_rollout_index"]
    )
    raw.update(failure_kind="verifier_error", failure_reason="preserved oracle incident", mask_sample=True)
    again = await ProteinLMBenchVerifier().verify(ProteinLMBenchRequest.model_validate(raw))
    assert again.failure_kind == "verifier_error" and again.mask_sample
    evidence = dict(raw, verifier_metadata={"sentence_index2aspects": {"0": ["a"]}, "limit": 1, "candidate_count": 1})
    scored = await Verifier().verify(Request.model_validate(evidence))
    assert scored.failure_kind == "verifier_error" and scored.mask_sample
    assert scored.model_dump()["_ng_task_index"] == raw["_ng_task_index"]
