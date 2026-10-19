# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym model-server agent loop for the three-stage GenoTEX pilot."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from fastapi import Body, Request
from pydantic import ConfigDict, Field, JsonValue, PrivateAttr

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.genotex.runtime import SHELL_TOOL, Container, score, stage, task_prompt


class GenoTEXConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    data_root: str
    work_root: str
    grader_python: str = "python3"
    image: str = "genotex-pilot:latest"
    max_concurrent: int = 1
    max_steps: int = 80
    timeout_s: int = 7200
    cpus: int = 4
    memory_gb: int = 16


class GenoTEXRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, JsonValue]


class GenoTEXResponse(BaseVerifyResponse):
    verifier_metadata: dict[str, JsonValue]
    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    artifact_dir: str
    elapsed_seconds: float


class GenoTEXAgent(SimpleResponsesAPIAgent):
    config: GenoTEXConfig
    _sem: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._sem = asyncio.Semaphore(self.config.max_concurrent)

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError("GenoTEX requires /run with a prepared task and sandbox")

    async def run(self, body: GenoTEXRunRequest = Body(), request: Request = None) -> GenoTEXResponse:
        async with self._sem:
            return await self._run(body, request)

    async def _run(self, body: GenoTEXRunRequest, request: Request | None) -> GenoTEXResponse:
        started = time.monotonic()
        task = body.verifier_metadata
        run_root = Path(self.config.work_root).expanduser().resolve() / uuid.uuid4().hex
        run_root.mkdir(parents=True)
        (run_root / "task.json").write_text(json.dumps(task, indent=2))
        response = NeMoGymResponse(
            id=run_root.name,
            created_at=time.time(),
            model="policy_model",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tools=[],
            tool_choice="auto",
        )
        metrics, failure, masked = {}, None, False
        container = None
        trajectory = []
        try:
            paths = stage(Path(self.config.data_root).expanduser().resolve(), run_root, task)
            container = Container(
                image=self.config.image, paths=paths, cpus=self.config.cpus, memory_gb=self.config.memory_gb
            )
            await container.start()
            (run_root / "container.json").write_text(json.dumps(await container.inspect(), indent=2))
            history = [{"role": "user", "content": task_prompt(task)}]
            params = body.responses_create_params.model_dump(exclude_none=True)
            params.update(tools=[SHELL_TOOL], parallel_tool_calls=False, temperature=0)
            async with asyncio.timeout(self.config.timeout_s):
                for step in range(self.config.max_steps):
                    reply = await self.server_client.post(
                        server_name=self.config.model_server.name,
                        url_path="/v1/responses",
                        json={**params, "input": history},
                        cookies=request.cookies if request else {},
                    )
                    await raise_for_status(reply)
                    payload = await get_response_json(reply)
                    response = NeMoGymResponse.model_validate(payload)
                    outputs = payload.get("output", [])
                    history.extend(outputs)
                    trajectory.append({"step": step, "response": payload})
                    calls = [item for item in outputs if item.get("type") == "function_call"]
                    if not calls:
                        if payload.get("status") == "incomplete":
                            reason = (payload.get("incomplete_details") or {}).get("reason", "unknown")
                            failure = f"model_response_incomplete:{reason}"
                        break
                    for call in calls:
                        try:
                            args = json.loads(call["arguments"])
                            if call["name"] != "shell" or not isinstance(args.get("command"), str):
                                raise ValueError("Expected shell(command: string)")
                            result = await container.shell(args["command"])
                        except (ValueError, TimeoutError) as exc:
                            result = {"error": str(exc)}
                        history.append(
                            {"type": "function_call_output", "call_id": call["call_id"], "output": json.dumps(result)}
                        )
                        trajectory.append({"step": step, "tool": call, "result": result})
                    (run_root / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
                else:
                    failure = "max_steps_exhausted"
            await container.stop()
            container = None
            metrics = await score(
                data_root=Path(self.config.data_root).expanduser().resolve(),
                run_root=run_root,
                paths=paths,
                task=task,
                python=self.config.grader_python,
            )
        except TimeoutError:
            failure = "task_timeout"
            try:
                if container:
                    await container.stop()
                    container = None
                metrics = await score(
                    data_root=Path(self.config.data_root).expanduser().resolve(),
                    run_root=run_root,
                    paths=paths,
                    task=task,
                    python=self.config.grader_python,
                )
            except Exception as exc:
                failure, masked = f"Grading after timeout failed: {type(exc).__name__}: {exc}", True
        except Exception as exc:
            # A task boundary: retain diagnostic and distinguish infrastructure failure.
            failure, masked = f"{type(exc).__name__}: {exc}", True
        finally:
            if container:
                try:
                    await container.stop()
                except Exception as exc:
                    failure, masked = f"Container cleanup failed: {type(exc).__name__}: {exc}", True
            (run_root / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
        result = GenoTEXResponse(
            **body.model_dump(),
            response=response,
            reward=float(metrics.get("reward", 0)),
            metrics=metrics,
            artifact_dir=str(run_root),
            elapsed_seconds=time.monotonic() - started,
            mask_sample=masked,
            failure_kind="genotex:infrastructure" if masked else None,
            failure_reason=failure,
        )
        (run_root / "result.json").write_text(result.model_dump_json(indent=2))
        return result


if __name__ == "__main__":
    GenoTEXAgent.run_webserver()
