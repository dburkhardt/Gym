# SPDX-License-Identifier: Apache-2.0
"""Join actual guarded native SSE receipts; retain uncertain role attribution."""

import hashlib
import json
from pathlib import Path

from openai.lib.streaming.chat._completions import ChatCompletionStreamState
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from nemo_gym.openai_utils import NeMoGymResponseUsage
from nemo_gym.rollout_observability import (
    AgentInvocation,
    ModelCallRef,
    ObservationGap,
    TrajectoryRecord,
    TrajectoryToolCall,
    TrajectoryTurn,
)


def completion(body, streaming):
    if not streaming:
        return ChatCompletion.model_validate_json(body).model_dump()
    events = [line[5:].strip() for line in body.splitlines() if line.startswith(b"data:")]
    if not events or events[-1] != b"[DONE]" or events.count(b"[DONE]") != 1:
        raise ValueError("Unresolved native SSE response")
    state = ChatCompletionStreamState()
    for event in events[:-1]:
        state.handle_chunk(ChatCompletionChunk.model_validate_json(event))
    return state.get_final_completion().model_dump()


def observations(directory: Path, trace, *, task_id, rollout_id, model_ref):
    trajectory = TrajectoryRecord(task_id=task_id, rollout_id=rollout_id)
    invocation = AgentInvocation(
        invocation_id="native",
        status="failed" if trace.get("errors") else "completed" if trace.get("is_completed") else "incomplete",
    )
    trajectory.invocations.append(invocation)
    trajectory.gaps.append(
        ObservationGap(
            code="native_role_attribution_unavailable",
            invocation_id="native",
            detail="Creator solver, helper and judge requests share transport; no inferred role labels.",
        )
    )
    requests = sorted(directory.glob("*.request.json"))
    usages = []
    for ordinal, path in enumerate(requests):
        if path.stem != f"{ordinal:06d}.request":
            raise ValueError("Noncontiguous native call requests")
        stem = path.with_name(f"{ordinal:06d}")
        if not stem.with_suffix(".complete.json").exists() or not stem.with_suffix(".intent.json").exists():
            trajectory.gaps.append(
                ObservationGap(code="native_call_receipt_incomplete", invocation_id="native", detail=path.name)
            )
            continue
        request = json.loads(path.read_text())
        intent = json.loads(stem.with_suffix(".intent.json").read_text())
        done = json.loads(stem.with_suffix(".complete.json").read_text())
        chunks = sorted(directory.glob(f"{ordinal:06d}.chunk*.json"))
        if any(p.name != f"{ordinal:06d}.chunk{n:06d}.json" for n, p in enumerate(chunks)):
            raise ValueError("Noncontiguous native SSE chunks")
        body = b"".join(bytes.fromhex(json.loads(p.read_text())["hex"]) for p in chunks)
        if (
            done.get("sha256") != hashlib.sha256(body).hexdigest()
            or done["bytes"] != len(body)
            or done["status"] != 200
        ):
            raise ValueError("Native response receipt hash/status mismatch")
        if intent["request_sha256"] != hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest():
            raise ValueError("Native request receipt hash mismatch")
        response = completion(body, request.get("stream", False))
        if response["model"] != request["model"] or len(response["choices"]) != 1:
            raise ValueError("Native completion identity changed")
        ref = ModelCallRef(model_ref=model_ref, response_id=response["id"])
        invocation.model_calls.append(ref)
        message = response["choices"][0]["message"]
        # Preserve native tool-call fields in the actual assistant message.
        trajectory.turns.append(
            TrajectoryTurn(
                invocation_id="native",
                task_id=task_id,
                rollout_id=rollout_id,
                turn_no=ordinal + 1,
                step_count=ordinal + 1,
                timestamp=intent["started_at"],
                question=request["messages"],
                answer=[message],
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
    for node in trace.get("nodes", []):
        message = node.get("message", {})
        if message.get("role") != "tool":
            continue
        if not message.get("tool_call_id"):
            trajectory.gaps.append(ObservationGap(code="native_tool_identity_missing", invocation_id="native"))
            continue
        trajectory.tool_calls.append(
            TrajectoryToolCall(
                invocation_id="native",
                tool_call_id=message["tool_call_id"],
                tool_name=message.get("name"),
                completed_at=node.get("timestamp"),
                timing_source="artifact",
                status="unknown",
                output=message.get("content"),
            )
        )
    if not requests:
        trajectory.gaps.append(ObservationGap(code="native_provider_receipts_unavailable", invocation_id="native"))
    return trajectory, NeMoGymResponseUsage.sum_from_list(usages) if requests and len(usages) == len(
        requests
    ) else None
