# SPDX-License-Identifier: Apache-2.0
"""Native stage observations from actual retained creator model receipts."""

import json
from pathlib import Path

from nemo_gym.openai_utils import NeMoGymResponseUsage
from nemo_gym.rollout_observability import (
    AgentInvocation,
    ModelCallRef,
    ObservationGap,
    TrajectoryRecord,
    TrajectoryTurn,
)


STAGES = ("inspiration_screening", "hypothesis_generation", "evaluate")


def observations(directory: Path, *, task_id, rollout_id, model_ref, imported=False):
    trajectory = TrajectoryRecord(task_id=task_id, rollout_id=rollout_id)
    records = [
        json.loads(line) for line in (directory / "model-traces.jsonl").read_text().splitlines() if line.strip()
    ]
    invocations, counts, usages, resolved = {}, {}, [], set()
    for record in records:
        stage = record["stage"]
        if stage not in STAGES:
            raise ValueError("Unknown native MOOSE-Chem stage")
        invocation = invocations.setdefault(stage, AgentInvocation(invocation_id=stage, status="completed"))
        response = record.get("response")
        if record.get("request_id"):
            if record["request_id"] in resolved:
                raise ValueError("Duplicate native request identity")
            resolved.add(record["request_id"])
        if not response:
            invocation.status = "failed"
            trajectory.gaps.append(
                ObservationGap(
                    code="failed_native_model_call", invocation_id=stage, detail=record.get("error_type", "unknown")
                )
            )
            continue
        if not response.get("id") or record.get("started") is None:
            trajectory.gaps.append(
                ObservationGap(code="native_call_identity_or_timestamp_missing", invocation_id=stage)
            )
            continue
        ref = ModelCallRef(model_ref=model_ref, response_id=response["id"])
        invocation.model_calls.append(ref)
        counts[stage] = counts.get(stage, 0) + 1
        message = response["choices"][0]["message"]
        trajectory.turns.append(
            TrajectoryTurn(
                invocation_id=stage,
                task_id=task_id,
                rollout_id=rollout_id,
                turn_no=counts[stage],
                step_count=counts[stage],
                timestamp=record["started"],
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
    unresolved = False
    intents = directory / "dispatch-intents.jsonl"
    if intents.exists():
        for line in intents.read_text().splitlines():
            intent = json.loads(line)
            if intent.get("request_id") not in resolved:
                unresolved = True
                if intent.get("stage") in invocations:
                    invocations[intent["stage"]].status = "incomplete"
                trajectory.gaps.append(
                    ObservationGap(
                        code="unresolved_native_model_intent",
                        invocation_id=intent.get("stage"),
                        detail=intent.get("request_id"),
                    )
                )
    if imported:
        trajectory.gaps.append(
            ObservationGap(
                code="imported_historical_execution",
                detail="Retained response IDs belong to historical captures, not newly dispatched calls.",
            )
        )
    trajectory.invocations = list(invocations.values())
    return trajectory, NeMoGymResponseUsage.sum_from_list(usages) if records and not unresolved and len(usages) == len(
        records
    ) else None
