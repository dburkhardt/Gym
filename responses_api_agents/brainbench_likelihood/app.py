# SPDX-License-Identifier: Apache-2.0
"""Two guarded whole-chat likelihood calls; not raw continuation likelihood."""

import asyncio
import hashlib
import json
import time
from pathlib import Path
from uuid import uuid4

from aiohttp import ClientTimeout
from fastapi import Request, Response
from pydantic import PrivateAttr

from benchmarks.brainbench.protocol import MODEL, mean_nll
from nemo_gym.base_responses_api_agent import Body
from nemo_gym.base_responses_api_model import CaptureStore
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseUsage
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


class BrainBenchAgent(NonExecutingSimpleAgent):
    config: Config
    _serial: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    async def responses(
        self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        async with self._serial:
            return await self._responses(request, response, body)

    async def _responses(
        self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        rollout_id = request.path_params.get("rollout_id")
        if not rollout_id or not self.server_client.global_config_dict.get("provider_dispatch_guard"):
            raise ValueError("BrainBench requires native rollout identity and provider_dispatch_guard")
        inputs = body.input
        if not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(inputs[0].content, str):
            raise ValueError("Expected one JSON pair message")
        pair = json.loads(inputs[0].content)
        if (
            set(pair) != {"choices"}
            or len(pair["choices"]) != 2
            or not all(isinstance(x, str) for x in pair["choices"])
        ):
            raise ValueError("Expected exactly two choice strings")
        # Exclusive attempt directory prevents an interrupted /responses restart from
        # inventing a fresh lineage. Completed native output resume happens upstream.
        # Reuse native dispatch evidence rather than treating a missing scorer
        # result as an unknown provider request. This campaign runs sequentially.
        previous_rollouts = []
        if self.config.output_root.exists():
            for previous in self.config.output_root.iterdir():
                if previous.is_dir():
                    identity = json.loads((previous / "attempt.json").read_text())
                    previous_rollouts.append(identity["rollout_id"])
        CaptureStore(
            Path(self.server_client.global_config_dict["model_call_capture_dir"]), provider_dispatch_guard=True
        ).assert_dispatch_reconciled(previous_rollouts)
        directory = self.config.output_root / hashlib.sha256(rollout_id.encode()).hexdigest()
        directory.mkdir(parents=True, exist_ok=False)
        lineage = uuid4().hex
        digest = configuration_digest(
            {"config": self.config.model_dump(mode="json"), "pair": pair, "protocol": "chat-mean-nll-v1"}
        )
        publish(directory / "attempt.json", {"rollout_id": rollout_id, "lineage": lineage, "digest": digest})
        receipts, values, usages = [], [], []
        trajectory = TrajectoryRecord(task_id=rollout_id, rollout_id=rollout_id)
        invocation = AgentInvocation(
            invocation_id="likelihood", status="completed", output_semantics="prompt_likelihood"
        )
        trajectory.invocations.append(invocation)
        for ordinal, text in enumerate(pair["choices"]):
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 1,
                "prompt_logprobs": 0,
                "stream": False,
            }
            started = time.time()
            headers = {
                "x-nemo-gym-logical-call-id": f"brainbench:{lineage}:{ordinal}",
                "x-nemo-gym-call-config-sha256": digest,
            }
            publish(
                directory / f"{ordinal}.intent.json", {"request": payload, "headers": headers, "started_at": started}
            )
            reply = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=self.url_path_for_request("/v1/chat/completions", request),
                json=payload,
                headers=headers,
                cookies=request.cookies,
                timeout=ClientTimeout(total=180),
            )
            await raise_for_status(reply)
            data = await get_response_json(reply)
            publish(directory / f"{ordinal}.response.json", data)
            values.append(mean_nll(data))
            receipts.append({"request": payload, "response": data})
            ref = ModelCallRef(model_ref=self.config.model_server, response_id=data["id"])
            invocation.model_calls.append(ref)
            trajectory.turns.append(
                TrajectoryTurn(
                    invocation_id="likelihood",
                    task_id=rollout_id,
                    rollout_id=rollout_id,
                    turn_no=ordinal + 1,
                    step_count=ordinal + 1,
                    timestamp=started,
                    question=payload["messages"],
                    answer=[
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": data["choices"][0]["message"].get("content") or ""}
                            ],
                        }
                    ],
                    model_calls=[ref],
                )
            )
            usage = data.get("usage") or {}
            if all(usage.get(k) is not None for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
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
            for key, value in reply.cookies.items():
                response.set_cookie(key, value.value)
        result = NeMoGymResponse.model_validate(
            {
                "usage": NeMoGymResponseUsage.sum_from_list(usages) if len(usages) == 2 else None,
                "id": "brainbench-" + lineage,
                "created_at": time.time(),
                "model": MODEL,
                "object": "response",
                "status": "completed",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "output": [
                    {
                        "id": "likelihood",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps({"mean_nll": values, "receipts": receipts}),
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "ng_trajectory": trajectory.model_dump(mode="json"),
            }
        )
        publish(directory / "result.json", result.model_dump(mode="json"))
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
    BrainBenchAgent.run_webserver()
