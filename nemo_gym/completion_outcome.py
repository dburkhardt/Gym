# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add explicit completion diagnostics without changing scientific scores or masks."""

from nemo_gym.openai_utils import NeMoGymResponse


def with_completion_outcome(result: dict, *, response: NeMoGymResponse) -> dict:
    """Preserve verifier/prior failure precedence; never infer exhaustion from usage."""
    updated = dict(result)
    if updated.get("failure_kind") or updated.get("failure_reason") or response.status != "incomplete":
        return updated
    if getattr(response, "error", None) is not None:
        updated["failure_kind"] = "response:provider_error"
        updated["failure_reason"] = "Incomplete response also contains provider error; raw error retained in response"
        return updated
    reason = getattr(response.incomplete_details, "reason", None)
    updated["failure_kind"] = "model_budget" if reason == "max_output_tokens" else "response:incomplete"
    updated["failure_reason"] = f"Provider returned incomplete response: {reason or 'unspecified'}"
    return updated
