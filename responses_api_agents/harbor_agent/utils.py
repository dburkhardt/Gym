# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import ValidationError

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)


@dataclass
class HarborAgentUtils:
    @staticmethod
    def validate_task_resources(task_path: Path, environment_config: object) -> None:
        """Reject implicit resource changes introduced by the Harbor pin upgrade."""
        from harbor.models.task.task import Task

        task = Task(task_path)
        environments = [("environment", task.config.environment)]
        verifiers = [("verifier", task.config.verifier)] + [
            (f"step {step.name} verifier", step.verifier) for step in task.config.steps or []
        ]
        for label, verifier in verifiers:
            if verifier is not None and verifier.environment is not None:
                environments.append((f"{label}.environment", verifier.environment))
        for label, resources in environments:
            missing = [
                field
                for field in ("cpus", "memory_mb")
                if getattr(resources, field) is None and getattr(environment_config, f"override_{field}", None) is None
            ]
            if missing:
                raise ValueError(
                    f"Task {task_path.name!r} needs explicit {label} {', '.join(missing)} before execution. "
                    "The previous Harbor pin defaulted to cpus=1 and memory_mb=2048. "
                    "Record the intended scientific resource limits in task.toml; do not silently uncap them."
                )

    @staticmethod
    def with_native_validity(trial_result: dict[str, Any]) -> dict[str, Any]:
        """Derive current Harbor policy without modifying the saved raw receipt.

        Historical receipts may not satisfy the current result schema; those
        continue through the explicit legacy classification in failure_diagnostics.
        """
        from harbor.models.trial.result import TrialResult

        try:
            parsed = TrialResult.model_validate(trial_result)
        except ValidationError:
            return trial_result
        return {
            **trial_result,
            "reward_valid": parsed.reward_valid,
            "failure_cause": parsed.failure_cause.value if parsed.failure_cause else None,
        }

    @staticmethod
    def _wrap_reasoning_in_think_tags(texts: List[str]) -> str:
        return "".join(f"<think>{text}</think>" for text in texts if text)

    @staticmethod
    def _merge_message_and_reasoning(message: str, reasoning_content: Optional[str]) -> str:
        """Merge reasoning back into the message text with ``<think>`` tags.

        The model's native output format is ``<think>reasoning</think>content``,
        so we prepend the wrapped reasoning before the message to preserve the
        original token order (important for training data alignment).
        """
        if not reasoning_content:
            return message
        wrapped_reasoning = HarborAgentUtils._wrap_reasoning_in_think_tags([reasoning_content])
        return f"{wrapped_reasoning}{message}"

    @staticmethod
    def get_default_response_object() -> Dict[str, Any]:
        return {
            "id": f"resp_{str(uuid4())}",
            "created_at": int(time.time()),
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {},
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "background": False,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "reasoning": {
                "effort": None,
                "generate_summary": None,
                "summary": None,
            },
            "service_tier": "default",
            "status": "completed",
            "text": {"format": {"type": "text"}, "verbosity": "medium"},
            "top_logprobs": 0,
            "truncation": "disabled",
            "usage": None,
            "user": None,
            "prompt_cache_key": None,
            "safety_identifier": None,
            "store": True,
        }

    @staticmethod
    def extract_reward(verifier_result: Optional[Dict[str, Any]]) -> float:
        """Extract reward from Harbor's VerifierResult.rewards dict.

        Harbor rewards are typically {"reward": 0.0 or 1.0} or a dict of named rewards.
        Returns the primary reward value, defaulting to 0.0 on failure.
        """
        if verifier_result is None:
            return 0.0

        rewards = verifier_result.get("rewards")
        if not rewards or not isinstance(rewards, dict):
            return 0.0

        # Return the "reward" key if present, otherwise return the first value
        if "reward" in rewards:
            return float(rewards["reward"])

        # Fallback: return first reward value
        for value in rewards.values():
            return float(value)

        return 0.0

    # ------------------------------------------------------------------ #
    #  Input extraction — populate responses_create_params.input          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def extract_input_from_trajectory(
        trajectory: Optional[Dict[str, Any]],
    ) -> List[NeMoGymEasyInputMessage]:
        """Extract the initial user instruction(s) from an ATIF trajectory.

        Harbor tasks provide the instruction via a file (not through the NeMo Gym
        request body).  The instruction appears as the first step(s) with
        ``source: "user"`` in the ATIF trajectory.  We convert these into
        ``NeMoGymEasyInputMessage`` dicts so they populate
        ``responses_create_params.input`` in the final output.

        Returns an empty list when no trajectory is available.
        """
        if not trajectory:
            return []

        input_messages: List[NeMoGymEasyInputMessage] = []
        for step in trajectory.get("steps", []):
            if step.get("source") == "user":
                input_messages.append(
                    NeMoGymEasyInputMessage(
                        role="user",
                        content=step.get("message", ""),
                        type="message",
                    )
                )
            else:
                # User messages always come first in ATIF; stop once we hit
                # the first non-user step.
                break

        return input_messages

    # ------------------------------------------------------------------ #
    #  Usage extraction                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def extract_usage(
        trial_result: Dict[str, Any],
        trajectory: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return a complete reported token pair, or unknown usage (None)."""
        return HarborAgentUtils.extract_usage_with_provenance(trial_result, trajectory)[0]

    @staticmethod
    def extract_usage_with_provenance(
        trial_result: Dict[str, Any],
        trajectory: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Prefer a complete ATIF pair; never combine counts from different sources.

        Zero is a reported value, not a fallback signal. Raw partial counts remain
        in provenance when the integer-valued response usage cannot represent them.
        """
        final_metrics = (trajectory or {}).get("final_metrics") or {}
        agent_result = trial_result.get("agent_result") or {}
        final_extra = final_metrics.get("extra") or {}
        agent_metadata = agent_result.get("metadata") or {}
        candidates = {
            "trajectory.final_metrics": {
                "input_tokens": final_metrics.get("total_prompt_tokens"),
                "output_tokens": final_metrics.get("total_completion_tokens"),
                "cached_tokens": final_metrics.get("total_cached_tokens"),
                "cache_write_tokens": final_extra.get("total_cache_write_tokens"),
                "reasoning_tokens": final_extra.get("total_reasoning_tokens"),
            },
            "trial_result.agent_result": {
                "input_tokens": agent_result.get("n_input_tokens"),
                "output_tokens": agent_result.get("n_output_tokens"),
                "cached_tokens": agent_result.get("n_cache_tokens"),
                "cache_write_tokens": agent_metadata.get("n_cache_write_tokens"),
                "reasoning_tokens": agent_metadata.get("n_reasoning_tokens"),
            },
        }

        def known_count(value: Any) -> bool:
            return isinstance(value, int) and not isinstance(value, bool) and value >= 0

        completeness = {
            "trajectory.final_metrics.extra": final_extra.get("provider_usage_complete"),
            "trial_result.agent_result.metadata": agent_metadata.get("provider_usage_complete"),
        }
        provenance = {"source": None, "reported_counts": candidates, "provider_usage_complete": completeness}
        if any(value is False for value in completeness.values()):
            return None, provenance
        for source, counts in candidates.items():
            if not all(known_count(counts[key]) for key in ("input_tokens", "output_tokens")):
                continue
            provenance["source"] = source
            cached = counts["cached_tokens"]
            return {
                "input_tokens": counts["input_tokens"],
                "input_tokens_details": {
                    "cached_tokens": cached if known_count(cached) else None,
                    "cache_write_tokens": counts["cache_write_tokens"]
                    if known_count(counts["cache_write_tokens"])
                    else None,
                },
                "output_tokens": counts["output_tokens"],
                "output_tokens_details": {
                    "reasoning_tokens": counts["reasoning_tokens"] if known_count(counts["reasoning_tokens"]) else None
                },
                "total_tokens": counts["input_tokens"] + counts["output_tokens"],
            }, provenance
        return None, provenance

    # ------------------------------------------------------------------ #
    #  Raw content parsing — extract function calls from raw LLM JSON     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        """Try to extract a JSON object from text that may have surrounding content.

        Handles the common case where terminus-2's raw LLM response is valid
        JSON, as well as responses with extra text before/after the JSON object.
        """
        # Fast path: try direct parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        # Slow path: find the first balanced {...} in the text
        if not text or "{" not in text:
            return None

        brace_depth = 0
        start = -1
        in_string = False
        escape_next = False

        for i, ch in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                if brace_depth == 0:
                    start = i
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0 and start >= 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    start = -1

        return None

    @staticmethod
    def _parse_raw_content_tool_calls(message: str, agent_step_index: int) -> List[Dict[str, Any]]:
        """Parse function calls from a raw terminus-2 JSON response.

        When ``raw_content=true`` in the trajectory config, the step message
        contains the full LLM JSON response (with ``analysis``, ``plan``,
        ``commands``, ``task_complete`` fields).  We extract the ``commands``
        array and ``task_complete`` flag to build ATIF-compatible tool_call
        dicts so that downstream processing can treat them identically to
        steps produced with ``raw_content=false``.
        """
        parsed = HarborAgentUtils._extract_json_object(message)
        if parsed is None:
            return []

        tool_calls: List[Dict[str, Any]] = []

        commands = parsed.get("commands", [])
        if isinstance(commands, list):
            for i, cmd in enumerate(commands):
                # Strict expected schema: each command is a dict with
                # "keystrokes" and optional "duration".
                if not isinstance(cmd, dict):
                    continue
                keystrokes = cmd.get("keystrokes")
                if not isinstance(keystrokes, str) or not keystrokes:
                    continue
                raw_duration = cmd.get("duration", 1.0)
                try:
                    duration = float(raw_duration)
                except (TypeError, ValueError):
                    duration = 1.0

                tool_calls.append(
                    {
                        "tool_call_id": f"call_{agent_step_index}_{i + 1}",
                        "function_name": "bash_command",
                        "arguments": {
                            "keystrokes": keystrokes,
                            "duration": duration,
                        },
                    }
                )

        return tool_calls

    # ------------------------------------------------------------------ #
    #  Output conversion — trajectory → NeMo Gym output items             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def trajectory_to_responses(trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert ATIF trajectory agent steps to NeMo Gym output items.

        Each agent step in the trajectory is converted to:
        1. An assistant **message** containing the agent's analysis/plan text.
           Uses ``NeMoGymResponseOutputMessageForTraining`` when the step
           carries token IDs / logprobs, otherwise ``NeMoGymResponseOutputMessage``.
        2. One **function_call** item per tool call the agent made.
        3. One **function_call_output** item per observation result.

        When the trajectory was produced with ``raw_content=true`` (no
        ``tool_calls`` on agent steps), function calls are parsed from the
        raw LLM JSON stored in ``step.message``.
        """
        output_items: List[Dict[str, Any]] = []
        agent_step_index = 0

        for step in trajectory.get("steps", []):
            if step.get("source") != "agent":
                continue

            text = HarborAgentUtils._merge_message_and_reasoning(
                step.get("message", "") or "",
                step.get("reasoning_content"),
            )
            content = [
                NeMoGymResponseOutputText(
                    annotations=[],
                    text=text,
                    type="output_text",
                    logprobs=None,
                ),
            ]

            # Use the training variant when token-level details are available
            # in the step metrics (written by NemoGymLLM / LiteLLM).
            metrics = step.get("metrics") or {}
            prompt_token_ids = metrics.get("prompt_token_ids")
            completion_token_ids = metrics.get("completion_token_ids")
            logprobs = metrics.get("logprobs")
            metrics_extra = metrics.get("extra") or {}
            if not isinstance(metrics_extra, dict):
                metrics_extra = {}
            routed_experts = metrics.get("routed_experts")
            if routed_experts is None:
                routed_experts = metrics_extra.get("routed_experts")
            has_token_details = prompt_token_ids or completion_token_ids or logprobs

            if has_token_details:
                message = NeMoGymResponseOutputMessageForTraining(
                    id=f"cht_{uuid4().hex[:12]}",
                    content=content,
                    role="assistant",
                    status="completed",
                    type="message",
                    prompt_token_ids=prompt_token_ids or [],
                    generation_token_ids=completion_token_ids or [],
                    generation_log_probs=logprobs or [],
                    routed_experts=routed_experts,
                )
            else:
                message = NeMoGymResponseOutputMessage(
                    id=f"cht_{uuid4().hex[:12]}",
                    content=content,
                    role="assistant",
                    status="completed",
                    type="message",
                )
            output_items.append(message.model_dump())

            tool_calls = step.get("tool_calls") or []
            # raw_content mode: parse function calls from the raw JSON message
            if not tool_calls:
                tool_calls = HarborAgentUtils._parse_raw_content_tool_calls(
                    step.get("message", "") or "", agent_step_index
                )

            observation = step.get("observation", {})
            results = observation.get("results", [])

            # --- Function calls ---
            for tc in tool_calls:
                arguments = tc.get("arguments", {})
                fc = NeMoGymResponseFunctionToolCall(
                    arguments=json.dumps(arguments) if isinstance(arguments, dict) else str(arguments),
                    call_id=tc.get("tool_call_id", f"call_{uuid4().hex[:8]}"),
                    name=tc.get("function_name", "unknown"),
                    type="function_call",
                    id=f"fc_{uuid4().hex[:8]}",
                    status="completed",
                )
                output_items.append(fc.model_dump())

            # --- Observation / function call outputs ---
            for i, result in enumerate(results):
                call_id = (
                    tool_calls[i].get("tool_call_id", f"call_{uuid4().hex[:8]}")
                    if i < len(tool_calls)
                    else f"call_{uuid4().hex[:8]}"
                )
                fco = NeMoGymFunctionCallOutput(
                    call_id=call_id,
                    output=result.get("content", ""),
                    type="function_call_output",
                    id=f"fco_{uuid4().hex[:8]}",
                    status="completed",
                )
                output_items.append(fco.model_dump())

            agent_step_index += 1

        return output_items

    # ------------------------------------------------------------------ #
    #  Main entry point — trial result → NeMo Gym output items            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def trial_result_to_responses(
        trial_result: Dict[str, Any],
        trajectory: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert Harbor trial output to NeMo Gym output items.

        All output is derived from the ATIF trajectory.  Token IDs and
        logprobs are read from each step's ``metrics`` (populated by
        ``NemoGymLLM``).  Returns an empty list when no trajectory is
        available.
        """
        if trajectory and trajectory.get("steps"):
            return HarborAgentUtils.trajectory_to_responses(trajectory)
        return []
