# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persist each actual creator model response in native Harbor ATIF format."""

import json
from pathlib import Path

from harbor.models.trajectories.trajectory import Trajectory

from benchmarks.ainsteinbench.native import MODEL
from nemo_gym.sandbox.lifecycle import _publish


def publish(directory: Path, messages: list, context):
    identity = json.loads((directory / "identity.json").read_text())
    responses = {
        int(p.name.split("-")[1].split(".")[0]): json.loads(p.read_text())
        for p in directory.glob("call-*.response.json")
    }
    steps = []
    ordinal = 0
    for message in messages:
        role = message["role"]
        if role == "tool":
            if not steps or steps[-1]["source"] != "agent":
                raise ValueError("Tool observation lacks native agent step")
            steps[-1].setdefault("observation", {"results": []})["results"].append(
                {"source_call_id": message["tool_call_id"], "content": message["content"]}
            )
            continue
        step = {
            "step_id": len(steps) + 1,
            "source": "agent" if role == "assistant" else role,
            "message": message.get("content") or "",
        }
        if role == "assistant":
            if ordinal not in responses:
                raise ValueError("Trajectory assistant step has no saved provider response")
            usage = responses[ordinal].get("usage") or {}
            step.update(
                model_name=MODEL,
                llm_call_count=1,
                extra={
                    "logical_call_id": f"ainstein:{identity['lineage']}:{ordinal}",
                    "provider_response_file": f"call-{ordinal}.response.json",
                    "provider_usage": usage,
                },
            )
            if message.get("reasoning_content"):
                step["reasoning_content"] = message["reasoning_content"]
            if message.get("tool_calls"):
                step["tool_calls"] = [
                    {
                        "tool_call_id": t["id"],
                        "function_name": t["function"]["name"],
                        "arguments": json.loads(t["function"]["arguments"]),
                    }
                    for t in message["tool_calls"]
                ]
            if usage:
                step["metrics"] = {
                    k: v
                    for k, v in {
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                    }.items()
                    if v is not None
                }
            ordinal += 1
        steps.append(step)
    if ordinal != len(responses):
        raise ValueError("Saved provider response missing from native trajectory")
    trajectory = {
        "schema_version": "ATIF-v1.8",
        "session_id": identity["lineage"],
        "agent": {
            "name": "ainstein-creator-four-tools",
            "version": "d9b1383e86c2ae43dcb3ddbcaf34c21ceb786cca",
            "model_name": MODEL,
        },
        "steps": steps,
        "notes": "One agent step per saved provider response; unresolved intents are retained separately.",
    }
    usages = [r.get("usage") or {} for r in responses.values()]
    if usages and all(all(u.get(k) is not None for k in ("prompt_tokens", "completion_tokens")) for u in usages):
        context.n_input_tokens = sum(u["prompt_tokens"] for u in usages)
        context.n_output_tokens = sum(u["completion_tokens"] for u in usages)
        cached = [(u.get("prompt_tokens_details") or {}).get("cached_tokens") for u in usages]
        context.n_cache_tokens = sum(cached) if all(x is not None for x in cached) else None
        trajectory["final_metrics"] = {
            "total_prompt_tokens": context.n_input_tokens,
            "total_completion_tokens": context.n_output_tokens,
        }
        if context.n_cache_tokens is not None:
            trajectory["final_metrics"]["total_cached_tokens"] = context.n_cache_tokens
    context.metadata = {
        "native_model_calls": ordinal,
        "provider_usage_complete": bool(usages)
        and all(u.get("prompt_tokens") is not None and u.get("completion_tokens") is not None for u in usages),
    }
    validated = Trajectory.model_validate(trajectory).model_dump(mode="json", exclude_none=True)
    _publish(directory.parent / "trajectory.json", validated)
    return validated
