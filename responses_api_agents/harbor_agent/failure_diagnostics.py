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
    if name == "ModelMissingSubmissionError":
        return "model_missing_submission"
    if name in {"NativeAgentIterationLimitError", "NativeAgentSurrenderError", "NativeIterationLimitError"}:
        return "harbor:agent_outcome_unscored"
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


def required_artifact_failure(manifest: object, sources: list[str]) -> tuple[str | None, str | None]:
    """Qualify explicitly required single-step captures, not optional debug artifacts."""
    if not sources:
        return None, None
    if not isinstance(manifest, list):
        return VERIFIER_ERROR, "Required artifact capture manifest is unavailable or malformed"
    for source in sources:
        entries = [entry for entry in manifest if isinstance(entry, Mapping) and entry.get("source") == source]
        if len(entries) != 1 or entries[0].get("status") not in {"ok", "empty"}:
            return VERIFIER_ERROR, f"Required artifact capture incomplete: {source}"
    return None, None


def intent_bound_provider_rejection(trial_result, *, trial_dir, capture_dir, rollout_id, endpoint):
    """Refine an internal HTTP500 only for the existing native call-intent format.

    Unsupported, malformed or ambiguous artifacts leave classification unchanged.
    This is reporting evidence, never retry authorization or capture reconciliation.
    The 60-record and byte caps bound this format reader, not scientific task limits;
    larger artifacts retain their original classification unchanged.
    """
    import json
    from pathlib import Path

    from nemo_gym.base_responses_api_model import CaptureStore

    error = trial_result.get("exception_info")
    if not isinstance(error, Mapping) or error.get("exception_type") != "ClientResponseError":
        return None
    message = error.get("exception_message")
    if not isinstance(message, str) or not message.startswith("500,") or not message.endswith(f"url='{endpoint}'"):
        return None
    if not isinstance(rollout_id, str) or not rollout_id or not capture_dir:
        return None
    try:
        native = Path(trial_dir) / "agent" / "native"
        if native.is_symlink() or native.parent.is_symlink() or not native.is_dir():
            return None
        intents = list(native.glob("call-*.intent.json"))
        if not 1 <= len(intents) <= 60:
            return None
        unanswered = [p for p in intents if not p.with_name(p.name.replace(".intent.json", ".response.json")).exists()]
        if len(unanswered) != 1 or unanswered[0].is_symlink() or unanswered[0].stat().st_size > 32 * 1024 * 1024:
            return None
        intent = json.loads(unanswered[0].read_text())
        headers = intent.get("headers", {})
        call = headers.get("x-nemo-gym-logical-call-id")
        config = headers.get("x-nemo-gym-call-config-sha256")
        if not isinstance(call, str) or not call or not isinstance(config, str) or not config:
            return None
        root = Path(capture_dir)
        if not root.is_dir():
            return None
        store = CaptureStore(root)
        if store.has_pending(rollout_id) or store.is_incomplete(rollout_id):
            return None
        path = store.path_for(rollout_id)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
            return None
        records = store.read(rollout_id)
        if not records or len(records) > 60:
            return None
        matches = [r for r in records if r.get("provider_binding", {}).get("logical_call") == call]
        if len(matches) != 1 or matches[0] is not records[-1]:
            return None
        record = matches[0]
        binding = record.get("provider_binding", {})
        if binding.get("logical_rollout") != rollout_id or binding.get("agent_config") != config:
            return None
        if not isinstance(intent.get("payload"), dict) or record.get("request") != intent["payload"]:
            return None
        if record.get("status_code") != 429 or record.get("error_category") != "rate_limit":
            return None
        attempts = record.get("provider_attempts")
        if not isinstance(attempts, list) or not attempts:
            return None
        ids = [a.get("attempt_id") for a in attempts if isinstance(a, Mapping)]
        if len(ids) != len(attempts) or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            return None
        if any(a.get("status_code") != 429 or a.get("outcome") != "rejected" for a in attempts):
            return None
        # Any other unsettled/failed call makes attribution ambiguous, even with no pending marker.
        for prior in records[:-1]:
            if prior.get("status_code") != 200 or prior.get("error_category") is not None:
                return None
            if any(a.get("outcome") not in {"captured", "rejected"} for a in prior.get("provider_attempts", [])):
                return None
        return (
            PROVIDER_QUOTA_EXHAUSTED,
            "Exact unanswered native call intent matches terminal captured HTTP429 rejection",
        )
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return None


def terminus_bound_provider_rejection(trial_result, *, trial_dir, capture_dir, rollout_id, endpoint):
    """Read the existing durable Terminus cursor; never authorize a retry."""
    import hashlib
    import json
    from pathlib import Path

    from nemo_gym.base_responses_api_model import CaptureStore
    from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import NemoGymLLM

    error = trial_result.get("exception_info")
    expected = (
        f"Server error '500 Internal Server Error' for url '{endpoint}'\n"
        "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500"
    )
    if not isinstance(error, Mapping) or error.get("exception_type") != "HTTPStatusError":
        return None
    if error.get("exception_message") != expected or not isinstance(rollout_id, str) or not rollout_id:
        return None
    if not capture_dir:
        return None
    try:
        agent = Path(trial_dir) / "agent"
        path = agent / "gym-call-state.json"
        if agent.is_symlink() or path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
            return None
        state = json.loads(path.read_text())
        NemoGymLLM._validate_dispatch_state(state)
        calls = list(state["calls"].values())
        if not 1 <= len(calls) <= 60 or calls[-1]["completed"] or any(not c["completed"] for c in calls[:-1]):
            return None
        root = Path(capture_dir)
        if not root.is_dir():
            return None
        store = CaptureStore(root)
        if store.has_pending(rollout_id) or store.is_incomplete(rollout_id):
            return None
        capture_path = store.path_for(rollout_id)
        if capture_path.is_symlink() or not capture_path.is_file() or capture_path.stat().st_size > 64 * 1024 * 1024:
            return None
        records = store.read(rollout_id)
        if len(records) != len(calls):
            return None
        seen_attempts = set()
        seen_calls = set()
        for index, (call, record) in enumerate(zip(calls, records)):
            binding = record.get("provider_binding", {})
            if binding.get("logical_rollout") != rollout_id or binding.get("logical_call") != call["logical_call_id"]:
                return None
            if binding.get("agent_config") != call["configuration_sha256"]:
                return None
            request = record.get("request")
            if (
                not isinstance(request, dict)
                or hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest() != call["request_sha256"]
            ):
                return None
            model_call = record.get("model_call_id")
            if not isinstance(model_call, str) or not model_call or model_call in seen_calls:
                return None
            seen_calls.add(model_call)
            attempts = record.get("provider_attempts")
            if not isinstance(attempts, list) or not attempts:
                return None
            final = index == len(calls) - 1
            if record.get("status_code") != (429 if final else 200) or record.get("error_category") != (
                "rate_limit" if final else None
            ):
                return None
            for n, attempt in enumerate(attempts):
                ident = attempt.get("attempt_id")
                if not isinstance(ident, str) or not ident or ident in seen_attempts:
                    return None
                seen_attempts.add(ident)
                success = not final and n == len(attempts) - 1
                if attempt.get("status_code") != (200 if success else 429) or attempt.get("outcome") != (
                    "captured" if success else "rejected"
                ):
                    return None
        return (
            PROVIDER_QUOTA_EXHAUSTED,
            "Exact durable Terminus call cursor matches terminal captured HTTP429 rejection",
        )
    except (OSError, ValueError, TypeError, AttributeError, KeyError, RuntimeError):
        return None
