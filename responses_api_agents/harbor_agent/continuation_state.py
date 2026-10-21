# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restricted JSON cursor for the guarded pinned Terminus completed-turn hook.

External tool snapshots and absolute task deadlines remain owned by Gym session
hooks. Native tmux is not supported: caller must stage an attested replacement
session before installing this cursor. There is no automatic run restart here.
"""

import copy
import hashlib
import inspect
import json
import math
import time
from dataclasses import dataclass

from pydantic import JsonValue


_CHAT_FIELDS = {
    "_messages",
    "_cumulative_input_tokens",
    "_cumulative_output_tokens",
    "_cumulative_cache_tokens",
    "_cumulative_cost",
    "_prompt_token_ids_list",
    "_completion_token_ids_list",
    "_logprobs_list",
    "_extra_list",
    "_interleaved_thinking",
    "_last_response_id",
}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _protocol(agent) -> str:
    from harbor.llms.chat import Chat

    options = agent.options.model_dump(mode="json")
    effective_kwargs = copy.deepcopy(agent._llm_call_kwargs)
    if agent._session_id_headers:
        headers = effective_kwargs.setdefault("extra_headers", {})
        if not isinstance(headers, dict):
            raise ValueError("Invalid effective model headers")
        for name in agent._session_id_headers:
            headers[name] = "<restored-native-session-id>"
    # Logs are supplied outside options. Preserve all native protocol options,
    # including model and turn budget. No heuristic stripping of nested fields.
    return _digest(
        {
            "options": options,
            "effective_call_kwargs_sha256": _digest(effective_kwargs),
            "chat_source_sha256": hashlib.sha256(inspect.getsource(Chat).encode()).hexdigest(),
            "parser_source_sha256": hashlib.sha256(inspect.getsource(type(agent._parser)).encode()).hexdigest(),
            "max_episodes": agent._max_episodes,
            "prompt": agent._prompt_template,
            "timeout_prompt": agent._timeout_template,
            "model_name": agent._model_name,
        }
    )


def capture_turn_cursor(
    agent, next_episode: int, next_prompt: str, *, binding: str, instruction: str, absolute_deadline: float
) -> dict[str, JsonValue]:
    """Capture JSON only at an already-attested completed turn; refuse unsupported state."""
    if agent.options.enable_summarize or agent._summarization_count or agent._subagent_rollout_details:
        raise ValueError("Summarized or delegated state cannot be continued")
    if agent._pending_subagent_refs or agent._pending_handoff_prompt:
        raise ValueError("Pending subagent state cannot be continued")
    if type(next_episode) is not int or not 0 < next_episode < agent._max_episodes:
        raise ValueError("Only nonterminal completed turns can be captured")
    if agent._n_episodes != next_episode or not isinstance(next_prompt, str) or not binding:
        raise ValueError("Turn cursor does not match the native loop")
    if not math.isfinite(absolute_deadline) or absolute_deadline <= time.time():
        raise ValueError("Original task deadline expired")
    chat = agent._chat
    if set(vars(chat)) != _CHAT_FIELDS | {"_model"}:
        raise ValueError("Pinned Chat state schema changed")
    state = {
        "version": 1,
        "binding": binding,
        "protocol": _protocol(agent),
        "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
        "absolute_deadline": absolute_deadline,
        "next_episode": next_episode,
        "next_prompt": next_prompt,
        "chat": {key: copy.deepcopy(getattr(chat, key)) for key in sorted(_CHAT_FIELDS)},
        "pending_completion": agent._pending_completion,
        "trajectory": [step.model_dump(mode="json") for step in agent._trajectory_steps],
        "api_request_times": list(agent._api_request_times),
        "session_id": agent._session_id,
        "last_response_model_name": getattr(agent, "_last_response_model_name", None),
    }
    # Roundtrip enforces portable JSON; never pickle model clients or tool handles.
    state = json.loads(json.dumps(state, allow_nan=False))
    return {"state": state, "sha256": _digest(state)}


@dataclass
class ValidatedTurnCursor:
    """Fully validated native state, staged before any live session is changed."""

    state: dict
    chat: object
    trajectory: list

    def install(self, agent, context) -> str:
        """Install after all sessions/tool snapshots have validated and staged successfully."""
        from harbor.agents.terminus_2.terminus_2 import SubagentMetrics

        state = self.state
        if time.time() >= state["absolute_deadline"]:
            raise ValueError("Original task deadline expired during staging")
        agent._gym_absolute_deadline = state["absolute_deadline"]
        context.n_input_tokens = self.chat.total_input_tokens
        context.n_output_tokens = self.chat.total_output_tokens
        context.n_cache_tokens = self.chat.total_cache_tokens
        context.cost_usd = self.chat.total_cost
        agent._chat = self.chat
        agent._trajectory_steps = self.trajectory
        agent._context = context
        agent._pending_completion = state["pending_completion"]
        agent._api_request_times = state["api_request_times"]
        agent._n_episodes = state["next_episode"]
        agent._gym_next_episode = state["next_episode"]
        agent._session_id = state["session_id"]
        agent._last_response_model_name = state["last_response_model_name"]
        agent._summarization_count = 0
        agent._subagent_metrics = SubagentMetrics()
        agent._subagent_rollout_details = []
        agent._pending_subagent_refs = None
        agent._pending_handoff_prompt = None
        if agent._session_id_headers:
            headers = agent._llm_call_kwargs.setdefault("extra_headers", {})
            headers.update({name: state["session_id"] for name in agent._session_id_headers})
        return state["next_prompt"]


def validate_turn_cursor(
    agent, cursor: dict[str, JsonValue], *, binding: str, instruction: str, absolute_deadline: float
) -> ValidatedTurnCursor:
    """Validate without modifying agent, workspace or deadline; changed budgets are refused."""
    from harbor.llms.chat import Chat
    from harbor.models.trajectories import Step

    if not isinstance(cursor, dict) or set(cursor) != {"state", "sha256"}:
        raise ValueError("Invalid cursor envelope")
    state = copy.deepcopy(cursor["state"])
    if not isinstance(state, dict) or cursor["sha256"] != _digest(state):
        raise ValueError("Cursor checksum mismatch")
    expected = {
        "version",
        "binding",
        "protocol",
        "instruction_sha256",
        "absolute_deadline",
        "next_episode",
        "next_prompt",
        "chat",
        "pending_completion",
        "trajectory",
        "api_request_times",
        "session_id",
        "last_response_model_name",
    }
    if set(state) != expected or state["version"] != 1:
        raise ValueError("Cursor schema mismatch")
    if state["binding"] != binding or state["protocol"] != _protocol(agent):
        raise ValueError("Cursor task/protocol binding mismatch")
    if state["instruction_sha256"] != hashlib.sha256(instruction.encode()).hexdigest():
        raise ValueError("Cursor instruction mismatch")
    if (
        state["absolute_deadline"] != absolute_deadline
        or not math.isfinite(absolute_deadline)
        or time.time() >= absolute_deadline
    ):
        raise ValueError("Original task deadline changed or expired")
    episode = state["next_episode"]
    if type(episode) is not int or not 0 < episode < agent._max_episodes:
        raise ValueError("Invalid continuation cursor")
    if type(state["pending_completion"]) is not bool or not isinstance(state["next_prompt"], str):
        raise ValueError("Invalid continuation state")
    if not isinstance(state["session_id"], str) or not state["session_id"]:
        raise ValueError("Invalid continuation session")
    chat_state = state["chat"]
    if not isinstance(chat_state, dict) or set(chat_state) != _CHAT_FIELDS:
        raise ValueError("Chat schema mismatch")
    messages = chat_state["_messages"]
    if not isinstance(messages, list) or len(messages) != 2 * episode:
        raise ValueError("Chat/cursor turn mismatch")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != ("user" if index % 2 == 0 else "assistant"):
            raise ValueError("Invalid Chat message sequence")
    for key in ("_cumulative_input_tokens", "_cumulative_output_tokens", "_cumulative_cache_tokens"):
        if type(chat_state[key]) is not int or chat_state[key] < 0:
            raise ValueError("Invalid cumulative token count")
    cost = chat_state["_cumulative_cost"]
    if type(cost) not in (float, int) or not math.isfinite(cost) or cost < 0:
        raise ValueError("Invalid cumulative cost")
    if (
        type(chat_state["_interleaved_thinking"]) is not bool
        or chat_state["_interleaved_thinking"] != agent.options.interleaved_thinking
    ):
        raise ValueError("Invalid reasoning mode")
    if chat_state["_last_response_id"] is not None and not isinstance(chat_state["_last_response_id"], str):
        raise ValueError("Invalid response chain")
    for key in ("_prompt_token_ids_list", "_completion_token_ids_list", "_logprobs_list", "_extra_list"):
        if not isinstance(chat_state[key], list):
            raise ValueError("Invalid rollout detail arrays")
    times = state["api_request_times"]
    if not isinstance(times, list) or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times):
        raise ValueError("Invalid request timing history")
    if state["last_response_model_name"] is not None and not isinstance(state["last_response_model_name"], str):
        raise ValueError("Invalid response model name")
    chat = Chat(agent._llm)
    for key, value in chat_state.items():
        setattr(chat, key, value)
    trajectory = [Step.model_validate(step) for step in state["trajectory"]]
    if len(trajectory) != episode + 1 or [step.source for step in trajectory] != ["user"] + ["agent"] * episode:
        raise ValueError("Trajectory does not match completed unsummarized turns")
    if [step.step_id for step in trajectory] != list(range(1, len(trajectory) + 1)):
        raise ValueError("Invalid trajectory step sequence")
    return ValidatedTurnCursor(state, chat, trajectory)
