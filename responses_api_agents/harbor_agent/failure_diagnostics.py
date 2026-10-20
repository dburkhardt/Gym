# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map Harbor measurement validity into Gym without changing native scores.

Legacy compatibility follows Harbor error_policy.py at 96ff9c6. Only native
agent budget/refusal outcomes may remain valid after an exception. Provider
admission budget failures are not benchmark budget exhaustion.
"""

import math
from collections.abc import Mapping

from nemo_gym.failure_kinds import (
    CANCELLED,
    PROVIDER_OOM_KILLED,
    PROVIDER_QUOTA_EXHAUSTED,
    PROVIDER_UNAVAILABLE,
    TRANSPORT_PEER_DROP,
    VERIFIER_ERROR,
)


_VALID_AGENT_ERRORS = frozenset(
    {
        "AgentTimeoutError",
        "ContextWindowExceededError",
        "OutputTokenExceededError",
        "ContextLengthExceededError",
        "OutputLengthExceededError",
        "AgentSafetyRefusalError",
        "ContentPolicyViolationError",
    }
)
_VERIFIER_ERRORS = frozenset(
    {
        "VerifierTimeoutError",
        "RewardFileNotFoundError",
        "RewardFileEmptyError",
        "VerifierOutputParseError",
    }
)


def _failure_kind(name: object, cause: object = None) -> str:
    if name in _VERIFIER_ERRORS or cause == "verifier_failed":
        return VERIFIER_ERROR
    if name in {"MemoryLimitExceededError", "SandboxLikelyOutOfMemoryError"}:
        return PROVIDER_OOM_KILLED
    if name in {"CancelledError"} or cause == "cancelled":
        return CANCELLED
    if name in {
        "ConnectionError",
        "APIConnectionError",
        "ApiConnectionError",
        "NetworkConnectionError",
        "ApiConnectionClosedError",
        "ApiResponseStalledError",
    }:
        return TRANSPORT_PEER_DROP
    if name in {"BudgetExceededError", "ApiUsageLimitError", "RateLimitError", "ApiRateLimitError"}:
        return PROVIDER_QUOTA_EXHAUSTED
    if cause == "sandbox_failed" or name in {
        "EnvironmentStartTimeoutError",
        "SandboxBuildFailedError",
        "AgentSetupTimeoutError",
        "HealthcheckError",
        "SSHAccessError",
        "GKEPodNotFoundError",
        "RuntimeRequestError",
    }:
        return PROVIDER_UNAVAILABLE
    if cause == "model_provider_failed":
        return "harbor:model_provider_failed"
    if cause == "harbor_failed":
        return "harbor:framework_failed"
    return "harbor:unclassified"


def measurement_failure(
    trial_result: Mapping[str, object],
    *,
    llm_failure: object = None,
) -> tuple[str | None, str | None]:
    """Return failure kind/reason; leave original rewards and exceptions untouched."""
    llm_error_name = llm_failure.get("exception_type") if isinstance(llm_failure, Mapping) else None
    if llm_failure and llm_error_name not in _VALID_AGENT_ERRORS:
        return _failure_kind(llm_error_name), f"LLM call failed: {llm_failure}"
    errors = []
    if isinstance(trial_result.get("exception_info"), Mapping):
        errors.append(trial_result["exception_info"])
    for step in trial_result.get("step_results") or []:
        if isinstance(step, Mapping) and isinstance(step.get("exception_info"), Mapping):
            errors.append(step["exception_info"])
    native_validity = trial_result.get("reward_valid")
    if native_validity is not None and not isinstance(native_validity, bool):
        return "harbor:invalid_result", "Invalid Harbor reward_valid metadata"
    if native_validity is False:
        cause = trial_result.get("failure_cause")
        kind = _failure_kind(errors[0].get("exception_type") if errors else None, cause)
        return kind, f"Harbor marks reward invalid: {cause or errors or 'unspecified cause'}"
    # On legacy records, only known agent-caused exceptions retain validity.
    if native_validity is None:
        for error in errors:
            name = error.get("exception_type")
            if name not in _VALID_AGENT_ERRORS:
                kind = _failure_kind(name)
                return kind, f"{name or 'HarborError'}: {error.get('exception_message', '')}"
    verifier = trial_result.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, Mapping) else None
    if not isinstance(rewards, Mapping) or not rewards:
        if errors or llm_error_name in _VALID_AGENT_ERRORS:
            # Pinned 0.1.42 job aggregation excludes absent verifier rewards;
            # never invent a zero or call an unrun verifier a verifier failure.
            return "harbor:agent_outcome_unscored", "Agent outcome has no verifier measurement"
        return VERIFIER_ERROR, "Harbor returned no verifier rewards"
    # The adapter chooses reward, otherwise the first native metric. Validate all
    # values because nonfinite values must not enter serialized metric output.
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in rewards.values()):
        return VERIFIER_ERROR, "Harbor returned invalid verifier rewards"
    return None, None
