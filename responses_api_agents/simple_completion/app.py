# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One literal completion through the guarded native model server."""

import hashlib
import json
import time
from pathlib import Path
from uuid import uuid4

from aiohttp import ClientTimeout
from fastapi import Request, Response
from pydantic import Field

from nemo_gym.openai_utils import (
    NeMoGymCompletion,
    NeMoGymCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.provider_dispatch import configuration_digest
from nemo_gym.rollout_observability import AgentInvocation, ModelCallRef, TrajectoryRecord, TrajectoryTurn
from nemo_gym.sandbox.guarded_chat_transport import publish
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.non_executing_simple_agent.app import (
    NonExecutingSimpleAgent,
    NonExecutingSimpleAgentConfig,
    NonExecutingSimpleAgentRunRequest,
    NonExecutingSimpleAgentVerifyResponse,
)


class Config(NonExecutingSimpleAgentConfig):
    output_root: Path
    model_name: str
    request_timeout_seconds: float = Field(default=600, gt=0, allow_inf_nan=False)


class SimpleCompletionAgent(NonExecutingSimpleAgent):
    config: Config

    async def responses(
        self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        rollout_id = request.path_params.get("rollout_id")
        if not rollout_id or not self.server_client.global_config_dict.get("provider_dispatch_guard"):
            raise ValueError("Literal completion requires native rollout identity and dispatch guard")
        if not isinstance(body.input, list) or len(body.input) != 1 or not isinstance(body.input[0].content, str):
            raise ValueError("Expected one serialized literal-completion request")
        payload = NeMoGymCompletionCreateParamsNonStreaming.model_validate(
            json.loads(body.input[0].content) | {"model": self.config.model_name}
        ).model_dump(exclude_unset=True)
        directory = self.config.output_root / hashlib.sha256(rollout_id.encode()).hexdigest()
        # Completed-output resume is supported by Gym. An interrupted attempt
        # cannot restart with a fresh lineage or claim arbitrary continuation.
        directory.mkdir(parents=True, exist_ok=False)
        lineage = uuid4().hex
        digest = configuration_digest({"config": self.config.model_dump(mode="json"), "request": payload})
        headers = {"x-nemo-gym-logical-call-id": f"completion:{lineage}:0", "x-nemo-gym-call-config-sha256": digest}
        started = time.time()
        publish(
            directory / "intent.json",
            {
                "rollout_id": rollout_id,
                "lineage": lineage,
                "request": payload,
                "headers": headers,
                "started_at": started,
            },
        )
        reply = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=self.url_path_for_request("/v1/completions", request),
            json=payload,
            headers=headers,
            cookies=request.cookies,
            timeout=ClientTimeout(total=self.config.request_timeout_seconds),
        )
        await raise_for_status(reply)
        data = await get_response_json(reply)
        publish(directory / "response.json", data)
        completion = NeMoGymCompletion.model_validate(data)
        if completion.model != self.config.model_name:
            raise ValueError("Completion model identity differs from configured model")
        choice = completion.choices[0]
        ref = ModelCallRef(model_ref=self.config.model_server, response_id=completion.id)
        output = [
            {
                "id": completion.id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": choice.text, "annotations": []}],
            }
        ]
        trajectory = TrajectoryRecord(
            task_id=rollout_id,
            rollout_id=rollout_id,
            invocations=[AgentInvocation(invocation_id="completion", status="completed", model_calls=[ref])],
            turns=[
                TrajectoryTurn(
                    invocation_id="completion",
                    task_id=rollout_id,
                    rollout_id=rollout_id,
                    turn_no=1,
                    step_count=1,
                    timestamp=started,
                    question=[{"type": "text", "text": payload["prompt"]}],
                    answer=output,
                    model_calls=[ref],
                )
            ],
        )
        usage = completion.usage or {}
        native_usage = None
        if all(
            type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            native_usage = {
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "input_tokens_details": {
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                },
                "output_tokens_details": {
                    "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                },
            }
        result = NeMoGymResponse.model_validate(
            {
                "id": completion.id,
                "created_at": completion.created,
                "model": completion.model,
                "object": "response",
                "status": "incomplete" if choice.finish_reason == "length" else "completed",
                "incomplete_details": {"reason": "max_output_tokens"} if choice.finish_reason == "length" else None,
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
                "output": output,
                "usage": native_usage,
                "ng_trajectory": trajectory.model_dump(mode="json"),
                "raw_completion": data,
                "raw_completion_request": payload,
            }
        )
        publish(directory / "result.json", result.model_dump(mode="json"))
        for key, value in reply.cookies.items():
            response.set_cookie(key, value.value)
        return result

    async def run(
        self, request: Request, body: NonExecutingSimpleAgentRunRequest
    ) -> NonExecutingSimpleAgentVerifyResponse:
        result = await super().run(request, body)
        result.ng_trajectory = result.response.model_dump().get("ng_trajectory")
        if result.ng_trajectory:
            task_id = str(body.model_dump().get("task_id", body.model_dump().get("uuid", "")))
            result.ng_trajectory["task_id"] = task_id
            for turn in result.ng_trajectory["turns"]:
                turn["task_id"] = task_id
        return result


if __name__ == "__main__":
    SimpleCompletionAgent.run_webserver()
