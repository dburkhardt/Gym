# SPDX-License-Identifier: Apache-2.0
"""Map retained creator SDK records into native Gym turns without rewriting them."""

import json
from pathlib import Path
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseUsage
from nemo_gym.rollout_observability import AgentInvocation, ModelCallRef, TrajectoryRecord, TrajectoryTurn


def observations(directory: Path, *, task_id: str, rollout_id: str, model_ref: ModelServerRef):
    trajectory = TrajectoryRecord(task_id=task_id, rollout_id=rollout_id)
    invocations = {}
    usages = []
    requests = sorted(directory.glob("*.request.json"))
    if not requests:
        raise ValueError("No persisted MetaSyn model calls")
    for ordinal, path in enumerate(requests):
        if path.name != f"{ordinal:06d}.request.json":
            raise ValueError("Noncontiguous MetaSyn model calls")
        request = json.loads(path.read_text())
        saved = json.loads(path.with_name(f"{ordinal:06d}.response.json").read_text())
        response = json.loads(saved["body"])
        if saved["status"] != 200 or len(response.get("choices", [])) != 1:
            raise ValueError("Unresolved MetaSyn model call")
        message = response["choices"][0]["message"]
        role = "solver" if ordinal == 0 else "judge"
        invocation = invocations.setdefault(
            role,
            AgentInvocation(
                invocation_id=role, parent_invocation_id=None if role == "solver" else "solver", status="completed"
            ),
        )
        ref = ModelCallRef(model_ref=model_ref, response_id=response["id"])
        invocation.model_calls.append(ref)
        step = 1 if role == "solver" else ordinal
        trajectory.turns.append(
            TrajectoryTurn(
                invocation_id=role,
                task_id=task_id,
                rollout_id=rollout_id,
                turn_no=step,
                step_count=step,
                timestamp=saved["started_at"],
                question=request["messages"],
                answer=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": message.get("content") or ""}],
                    }
                ],
                reasoning_content=message.get("reasoning_content"),
                model_calls=[ref],
            )
        )
        usage = response.get("usage")
        if usage and all(usage.get(k) is not None for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
            usages.append(
                NeMoGymResponseUsage(
                    input_tokens=usage["prompt_tokens"],
                    output_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"],
                    input_tokens_details={
                        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                    },
                    output_tokens_details={
                        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                    },
                )
            )
    trajectory.invocations = list(invocations.values())
    return trajectory, NeMoGymResponseUsage.sum_from_list(usages) if len(usages) == len(requests) else None
