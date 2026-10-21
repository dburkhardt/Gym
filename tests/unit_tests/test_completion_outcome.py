# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from nemo_gym.completion_outcome import with_completion_outcome


@pytest.mark.parametrize(
    "reason,kind",
    [
        ("max_output_tokens", "model_budget"),
        ("content_filter", "response:incomplete"),
        (None, "response:incomplete"),
    ],
)
def test_explicit_reason(reason, kind):
    old = {"reward": 0.0, "mask_sample": False}
    new = with_completion_outcome(
        old, response=SimpleNamespace(status="incomplete", incomplete_details=SimpleNamespace(reason=reason))
    )
    assert new["failure_kind"] == kind and new["reward"] == 0 and new["mask_sample"] is False
    assert "failure_kind" not in old


def test_prior_failure():
    old = {"reward": 0.0, "mask_sample": True, "failure_kind": "verifier_error", "failure_reason": "oracle"}
    assert with_completion_outcome(old, response=SimpleNamespace(status="incomplete")) == old


def test_complete():
    old = {"reward": 1.0, "mask_sample": False}
    assert (
        with_completion_outcome(old, response=SimpleNamespace(status="completed", usage={"output_tokens": 16384}))
        == old
    )


def test_provider_error_prevents_budget_attribution():
    error = {"code": "server_error", "message": "ambiguous upstream"}
    response = SimpleNamespace(
        status="incomplete", incomplete_details=SimpleNamespace(reason="max_output_tokens"), error=error
    )
    result = with_completion_outcome({"reward": 0.0, "mask_sample": True}, response=response)
    assert result["failure_kind"] == "response:provider_error"
    assert result["mask_sample"] is True and response.error == error
