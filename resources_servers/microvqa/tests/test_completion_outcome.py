# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path

import pytest

from resources_servers.microvqa.app import Request, Verifier


@pytest.mark.asyncio
async def test_saved45_completion_and_scientific_parity():
    base = Path(os.environ.get("MICROVQA_FOUNDATION", "missing"))
    paths = [
        base / "microvqa-saved20/regraded-v2.jsonl",
        base / "microvqa-saved25/original17/regraded.jsonl",
        base / "microvqa-saved25/recovery8/regraded.jsonl",
    ]
    if not all(path.exists() for path in paths):
        pytest.skip("Actual saved45 cohorts required")
    count = incomplete = 0
    for path in paths:
        for line in path.read_text().splitlines():
            original = json.loads(line)
            result = (await Verifier().verify(Request.model_validate(original))).model_dump()
            for field in (
                "responses_create_params",
                "response",
                "verifier_metadata",
                "reward",
                "mask_sample",
                "diagnostics",
                "_ng_task_index",
                "_ng_rollout_index",
            ):
                assert result[field] == original[field], (path, count, field)
            if original["response"]["status"] == "incomplete":
                incomplete += 1
                if not original.get("failure_kind") and not original.get("failure_reason"):
                    reason = (original["response"].get("incomplete_details") or {}).get("reason")
                    expected = (
                        "response:provider_error"
                        if original["response"].get("error")
                        else "model_budget"
                        if reason == "max_output_tokens"
                        else "response:incomplete"
                    )
                    assert result["failure_kind"] == expected
            count += 1
    assert count == 45
    assert incomplete == 0  # Actual preserved45 are all completed; do not invent a budget observation.
    original["response"]["status"] = "incomplete"
    original["response"]["incomplete_details"] = {"reason": "max_output_tokens"}
    original["response"]["error"] = None
    original["failure_kind"] = None
    original["failure_reason"] = None
    variant = await Verifier().verify(Request.model_validate(original))
    assert variant.failure_kind == "model_budget"
    assert variant.reward == original["reward"] and variant.mask_sample == original["mask_sample"]
    original.update(mask_sample=True, failure_kind="verifier_error", failure_reason="preserved image incident")
    result = await Verifier().verify(Request.model_validate(original))
    assert result.mask_sample and result.failure_kind == "verifier_error"
    assert result.failure_reason == "preserved image incident"
