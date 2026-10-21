# SPDX-License-Identifier: Apache-2.0
"""Project persisted creator SDK decisions into Gym's native trajectory schema."""

import json
from pathlib import Path

from nemo_gym.config_types import ModelServerRef
from nemo_gym.rollout_observability import AgentInvocation, ModelCallRef, TrajectoryRecord, TrajectoryTurn


def build_trajectory(directory: Path, *, task_id: str, rollout_id: str, model_ref: ModelServerRef) -> TrajectoryRecord:
    trajectory = TrajectoryRecord(task_id=task_id, rollout_id=rollout_id)
    invocation = AgentInvocation(invocation_id="root", status="completed")
    requests = sorted(directory.glob("*.request.json"))
    if not requests:
        raise ValueError("No persisted creator decisions")
    for index, request_path in enumerate(requests):
        if request_path.name != f"{index:06d}.request.json":
            raise ValueError("Non-contiguous creator decisions")
        request = json.loads(request_path.read_text())
        saved = json.loads(request_path.with_name(f"{index:06d}.response.json").read_text())
        response = json.loads(saved["body"])
        if saved["status"] != 200 or len(response["choices"]) != 1:
            raise ValueError("Unresolved creator decision")
        choice = response["choices"][0]
        message = choice["message"]
        refs = [ModelCallRef(model_ref=model_ref, response_id=response["id"])]
        answer = []
        if message.get("content"):
            answer.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": message["content"]}],
                }
            )
        for call in message.get("tool_calls") or []:
            answer.append(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                }
            )
        trajectory.turns.append(
            TrajectoryTurn(
                invocation_id="root",
                task_id=task_id,
                rollout_id=rollout_id,
                turn_no=index + 1,
                timestamp=saved["started_at"],
                question=request["messages"],
                answer=answer,
                reasoning_content=message.get("reasoning_content"),
                step_count=index + 1,
                model_calls=refs,
            )
        )
        invocation.model_calls.extend(refs)
    trajectory.invocations.append(invocation)
    return trajectory
