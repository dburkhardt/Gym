# SPDX-License-Identifier: Apache-2.0
"""Map persisted native calls, not terminal summaries, into Gym observations."""

import hashlib
import json
import re
from pathlib import Path

from nemo_gym.openai_utils import NeMoGymResponseUsage
from nemo_gym.rollout_observability import (
    AgentInvocation,
    ModelCallRef,
    ObservationGap,
    TrajectoryRecord,
    TrajectoryToolCall,
    TrajectoryTurn,
)


def visible_tool_output(url, body):
    """Mirror the native loop's byte truncation and BLAST RID presentation."""
    tool = body
    if "blast" in url and "Put" in url:
        match = re.search("RID = (.*)\n", body.decode())
        if not match:
            return None
        tool = match.group(1)
    return str(tool[:20000])


def observations(directory: Path, *, task_id, rollout_id, model_ref, status):
    trajectory = TrajectoryRecord(task_id=task_id, rollout_id=rollout_id)
    invocation = AgentInvocation(
        invocation_id="solver",
        status="completed"
        if status == "completed"
        else "incomplete"
        if status in {"model_budget", "step_budget"}
        else "failed",
        stop_reason="max_iterations" if status == "step_budget" else status if status != "completed" else None,
    )
    trajectory.invocations.append(invocation)
    records = sorted(directory.glob("model-[0-9][0-9].json"))
    usages = []
    for ordinal, path in enumerate(records):
        record = json.loads(path.read_text())
        if record["turn"] != ordinal:
            raise ValueError("Noncontiguous native GeneHop model receipts")
        response = record["response"]
        if model_ref is None or not response.get("id") or "started_at" not in record:
            trajectory.gaps.append(
                ObservationGap(
                    code="native_call_identity_or_timestamp_missing", invocation_id="solver", detail=path.name
                )
            )
            continue
        ref = ModelCallRef(model_ref=model_ref, response_id=response["id"])
        invocation.model_calls.append(ref)
        message = response["choices"][0]["message"]
        trajectory.turns.append(
            TrajectoryTurn(
                invocation_id="solver",
                task_id=task_id,
                rollout_id=rollout_id,
                turn_no=ordinal + 1,
                step_count=ordinal + 1,
                timestamp=record["started_at"],
                question=record["request"]["messages"],
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
    for path in sorted(directory.glob("tool-*.json")):
        receipt = json.loads(path.read_text())
        body = path.with_suffix(".body").read_bytes()
        if hashlib.sha256(body).hexdigest() != receipt["sha256"]:
            raise ValueError("Native tool body hash mismatch")
        trajectory.tool_calls.append(
            TrajectoryToolCall(
                invocation_id="solver",
                tool_call_id=path.stem,
                tool_name=receipt["url"],
                started_at=receipt.get("started_at"),
                completed_at=receipt["time"],
                timing_source="artifact",
                status="completed",
                output=visible_tool_output(receipt["url"], body),
            )
        )
    for intent in sorted(directory.glob("model-*-intent.json")):
        completed = intent.with_name(intent.name.replace("-intent", ""))
        if not completed.exists():
            trajectory.gaps.append(
                ObservationGap(code="unresolved_native_model_intent", invocation_id="solver", detail=intent.name)
            )
    if not records:
        trajectory.gaps.append(ObservationGap(code="no_native_model_receipts", invocation_id="solver"))
    return trajectory, NeMoGymResponseUsage.sum_from_list(usages) if records and len(usages) == len(records) else None
