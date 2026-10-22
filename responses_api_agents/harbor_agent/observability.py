# SPDX-License-Identifier: Apache-2.0
"""Bind guarded Terminus solver calls using durable dispatch identity.

This reports ownership, not a replacement ATIF trajectory. The common collector
projects turns from captured responses only after these explicit bindings exist.
"""

import hashlib
import json

from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ModelCallRef


def completed_terminus_observations(state, captures, *, rollout_id):
    """Return proven completed solver bindings; reject partial or mixed histories.

    Callers must retain unavailable observations on rejection. No inference or
    retry is performed, and the invocation outcome remains unknown.
    """
    from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import NemoGymLLM

    NemoGymLLM._validate_dispatch_state(state)
    calls = list(state["calls"].values())
    if not rollout_id or not calls or len(calls) != len(captures):
        raise ValueError("Incomplete Terminus capture set")
    refs = []
    seen = set()
    for call, capture in zip(calls, captures):
        binding = capture.get("provider_binding") or {}
        request = capture.get("request")
        ident = capture.get("model_call_id")
        if (
            not call["completed"]
            or binding.get("logical_rollout") != rollout_id
            or binding.get("logical_call") != call["logical_call_id"]
            or binding.get("agent_config") != call["configuration_sha256"]
            or not isinstance(request, dict)
            or hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest() != call["request_sha256"]
            or not isinstance(ident, str)
            or not ident
            or ident in seen
            or capture.get("status_code") != 200
            or capture.get("error_category") is not None
            or not isinstance(capture.get("response"), dict)
        ):
            raise ValueError("Unproven Terminus solver capture binding")
        seen.add(ident)
        refs.append(ModelCallRef(model_call_id=ident))
    return AgentObservationBundle(
        source="harbor:terminus_dispatch_cursor",
        records=[AgentInvocation(invocation_id="solver", model_calls=refs)],
    )


def read_terminus_observations(*, trial_dir, capture_dir, rollout_id):
    """Optional reporting must never change the scientific result on missing evidence."""
    from pathlib import Path

    from nemo_gym.rollout_observability import ObservationGap

    try:
        from nemo_gym.base_responses_api_model import CaptureStore

        if not capture_dir or not rollout_id:
            raise ValueError("Capture configuration unavailable")
        agent = Path(trial_dir) / "agent"
        cursor = agent / "gym-call-state.json"
        if (
            agent.is_symlink()
            or cursor.is_symlink()
            or not cursor.is_file()
            or cursor.stat().st_size > 32 * 1024 * 1024
        ):
            raise ValueError("Dispatch cursor unavailable")
        root = Path(capture_dir)
        if not root.is_dir() or root.is_symlink():
            raise ValueError("Capture directory unavailable")
        store = CaptureStore(root)
        path = store.path_for(rollout_id)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Capture file unavailable")
        if store.has_pending(rollout_id) or store.is_incomplete(rollout_id):
            raise ValueError("Capture settlement incomplete")
        return completed_terminus_observations(
            json.loads(cursor.read_text()), store.read(rollout_id), rollout_id=rollout_id
        )
    except Exception as exc:
        # Reporting is optional: retain an explicit gap, never replace a native
        # scientific outcome. BaseException cancellation still propagates.
        return AgentObservationBundle(
            source="harbor:terminus_dispatch_cursor",
            gaps=[ObservationGap(code="harbor_solver_call_bindings_unavailable", detail=type(exc).__name__)],
        )
