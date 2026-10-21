# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.rollout_reverification import InputRolloutPair, _build_verify_payload
from resources_servers.campaign_creator_scoring import app


@pytest.mark.asyncio
async def test_pinned_scorer_provider_error_and_prior_incident():
    path = Path("../../outputs/broad-eval-campaign/foundation/qcbench-native-validity-regrade50-v1/results.jsonl")
    source = Path("../qcbench-upstream").resolve()
    if not path.exists() or not source.exists():
        pytest.skip("Pinned saved QCBench required")
    raw = next(json.loads(line) for line in path.read_text().splitlines() if json.loads(line)["uuid"] == "294")
    raw = copy.deepcopy(raw)
    raw.update(failure_kind=None, failure_reason=None)
    raw["response"].update(
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
        error={"code": "server_error", "message": "ambiguous provider error"},
    )
    server = SimpleNamespace(
        config=SimpleNamespace(benchmark="qcbench", upstream_repo=str(source), scoring_timeout_s=30)
    )
    result = await app.CreatorResourcesServer.verify(server, app.VerifyRequest.model_validate(raw))
    assert result.reward == raw["reward"] and result.mask_sample == raw["mask_sample"]
    assert result.failure_kind == "response:provider_error"
    assert result.response.error.message == "ambiguous provider error"
    assert result.scoring_details == raw["scoring_details"]
    raw.update(mask_sample=True, failure_kind="verifier_error", failure_reason="explicit independent incident")
    again = await app.CreatorResourcesServer.verify(server, app.VerifyRequest.model_validate(raw))
    assert again.mask_sample and again.failure_kind == "verifier_error"
    assert again.failure_reason == "explicit independent incident"
    raw.pop("mask_sample")
    raw["failure_kind"] = "model_budget"
    raw["failure_reason"] = "explicit prior generation evidence"
    raw["response"].update(status="completed", incomplete_details=None, error=None)
    failure_only = await app.CreatorResourcesServer.verify(server, app.VerifyRequest.model_validate(raw))
    assert failure_only.mask_sample is False
    assert failure_only.failure_kind == "model_budget"
    assert failure_only.failure_reason == "explicit prior generation evidence"


def test_native_reverify_does_not_carry_old_judge_failure():
    pair = InputRolloutPair(
        input={"uuid": "x", "verifier_metadata": {}},
        rollout={"response": {}, "mask_sample": True, "failure_kind": "judge_failed", "failure_reason": "old verdict"},
    )
    payload = _build_verify_payload(pair)
    assert not {"failure_kind", "failure_reason", "mask_sample"} & payload.keys()
