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
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.provider_dispatch import configuration_digest
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.genotex.checkpoint import remaining_budget, restore_checkpoint, save_checkpoint
from responses_api_agents.genotex.runtime import SHELL_TOOL, Container, execution_identity, score, stage, task_prompt


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
    model_config = ConfigDict(extra="allow")

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
        task = {k: v for k, v in body.verifier_metadata.items() if k not in ("resume_checkpoint", "resume_resolution")}
        work_root = Path(self.config.work_root).expanduser().resolve()
        resume_path = body.verifier_metadata.get("resume_checkpoint")
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
        state = None
        prior_elapsed = 0.0
        checkpoint_path = run_root / "checkpoint.json"

        def persist():
            state["elapsed_seconds"] = prior_elapsed + time.monotonic() - started
            save_checkpoint(checkpoint_path, state)

        try:
            if resume_path:
                if get_global_config_dict().get("provider_dispatch_guard"):
                    selected_checkpoint = Path(str(resume_path)).resolve()
                    if not selected_checkpoint.is_relative_to(work_root):
                        raise ValueError("Checkpoint outside configured work root")
                    saved_state = json.loads(selected_checkpoint.read_text())
                    if (
                        saved_state.get("pending")
                        and (body.verifier_metadata.get("resume_resolution") or {}).get("outcome") != "settled_error"
                    ):
                        raise ValueError(
                            "Guarded pending action requires durable ledger reconciliation; caller assertion is insufficient"
                        )
                state = restore_checkpoint(
                    Path(str(resume_path)),
                    work_root=work_root,
                    task=task,
                    resolution=body.verifier_metadata.get("resume_resolution"),
                )
                prior_elapsed = state["elapsed_seconds"]
                paths = {k: Path(v) for k, v in state["paths"].items()}
                (run_root / "resumed-from.json").write_text(json.dumps({"checkpoint": str(resume_path)}))
            else:
                paths = stage(Path(self.config.data_root).expanduser().resolve(), run_root, task)
                state = {
                    "version": 1,
                    "task": task,
                    "paths": {k: str(v) for k, v in paths.items()},
                    "history": [{"role": "user", "content": task_prompt(task)}],
                    "trajectory": [],
                    "next_step": 0,
                    "elapsed_seconds": 0.0,
                    "pending": None,
                    "pending_calls": [],
                    "solver_done": False,
                }
            remaining_s, _ = remaining_budget(state, timeout_s=self.config.timeout_s, max_steps=self.config.max_steps)
            history, trajectory = state["history"], state["trajectory"]
            if state.get("last_response"):
                response = NeMoGymResponse.model_validate(state["last_response"])
            persist()
            container = Container(
                image=self.config.image,
                paths=paths,
                cpus=self.config.cpus,
                memory_gb=self.config.memory_gb,
                ttl_s=remaining_s,
            )
            await container.start()
            observed_container = await container.inspect()
            (run_root / "container.json").write_text(json.dumps(observed_container, indent=2))
            observed_runtime = await execution_identity(
                image=observed_container,
                grader_python=self.config.grader_python,
                data_root=Path(self.config.data_root).expanduser().resolve(),
            )
            save_checkpoint(run_root / "runtime-identity.json", observed_runtime)
            if resume_path and state.get("runtime_identity") != observed_runtime:
                raise ValueError("Resume requires matching recorded runtime and resolved image identity")
            state["runtime_identity"] = observed_runtime
            persist()
            params = body.responses_create_params.model_dump(exclude_none=True)
            params.update(tools=[SHELL_TOOL], parallel_tool_calls=False, temperature=0)
            if (
                resume_path
                and (body.verifier_metadata.get("resume_resolution") or {}).get("outcome") == "settled_error"
            ):
                retained = state["settled_model_errors"][-1]["request"]
                expected = {**params, "input": history}
                # The gateway wire adds model at the model-server boundary.
                expected["model"] = get_global_config_dict().get("policy_model_name")
                if retained != expected:
                    raise ValueError("Settled-error continuation request/configuration differs")
            guarded = bool(get_global_config_dict().get("provider_dispatch_guard"))
            dispatch_digest = configuration_digest(
                {
                    **self.config.model_dump(),
                    "task": task,
                    "params": params,
                    "runtime": observed_runtime,
                    "policy_model_name": get_global_config_dict().get("policy_model_name"),
                }
            )
            if guarded:
                previous_digest = state.get("guard_config_sha256")
                if resume_path and previous_digest != dispatch_digest:
                    raise ValueError("Guarded resume requires matching native checkpoint/configuration")
                state["guard_config_sha256"] = dispatch_digest
                persist()

            remaining_s = self.config.timeout_s - prior_elapsed - (time.monotonic() - started)
            if remaining_s <= 0:
                raise TimeoutError("Original elapsed budget exhausted during setup")
            async with asyncio.timeout(remaining_s):
                while not state["solver_done"]:
                    if not state["pending_calls"]:
                        if state["next_step"] >= self.config.max_steps:
                            failure = "max_steps_exhausted"
                            break
                        step = state["next_step"]
                        state["pending"] = {"kind": "model", "action_id": uuid.uuid4().hex, "step": step}
                        persist()
                        reply = await self.server_client.post(
                            server_name=self.config.model_server.name,
                            url_path="/v1/responses",
                            json={**params, "input": history},
                            cookies=request.cookies if request else {},
                            headers={
                                "x-nemo-gym-logical-call-id": "genotex:" + state["pending"]["action_id"],
                                "x-nemo-gym-call-config-sha256": dispatch_digest,
                            }
                            if guarded
                            else {},
                            _max_num_tries=1,
                            _max_connection_retries=0,
                            _resend_after_disconnect=False,
                        )
                        await raise_for_status(reply)
                        payload = await get_response_json(reply)
                        response = NeMoGymResponse.model_validate(payload)
                        state["last_response"] = payload
                        outputs = payload.get("output", [])
                        history.extend(outputs)
                        trajectory.append({"step": step, "response": payload})
                        state["pending_calls"] = [item for item in outputs if item.get("type") == "function_call"]
                        state["next_step"] += 1
                        state["pending"] = None
                        if not state["pending_calls"]:
                            state["solver_done"] = True
                            if payload.get("status") == "incomplete":
                                reason = (payload.get("incomplete_details") or {}).get("reason", "unknown")
                                failure = f"model_response_incomplete:{reason}"
                        persist()
                    while state["pending_calls"]:
                        call = state["pending_calls"][0]
                        action_id = uuid.uuid4().hex
                        state["pending"] = {"kind": "tool", "action_id": action_id, "call": call}
                        persist()
                        try:
                            args = json.loads(call["arguments"])
                            if call["name"] != "shell" or not isinstance(args.get("command"), str):
                                raise ValueError("Expected shell(command: string)")
                            result = await container.shell(args["command"])
                        except (ValueError, TimeoutError) as exc:
                            result = {"error": str(exc)}
                        receipt = {"action_id": action_id, "call": call, "result": result}
                        save_checkpoint(run_root / f"tool-{action_id}.json", receipt)
                        history.append(
                            {"type": "function_call_output", "call_id": call["call_id"], "output": json.dumps(result)}
                        )
                        trajectory.append({"step": state["next_step"] - 1, "tool": call, "result": result})
                        state["pending_calls"].pop(0)
                        state["pending"] = None
                        persist()
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
            if state is not None:
                persist()
            (run_root / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
        result = GenoTEXResponse(
            **body.model_dump(),
            response=response,
            reward=float(metrics.get("reward", 0)),
            metrics=metrics,
            artifact_dir=str(run_root),
            elapsed_seconds=prior_elapsed + time.monotonic() - started,
            mask_sample=masked,
            failure_kind="genotex:infrastructure" if masked else None,
            failure_reason=failure,
        )
        (run_root / "result.json").write_text(result.model_dump_json(indent=2))
        return result


if __name__ == "__main__":
    GenoTEXAgent.run_webserver()
