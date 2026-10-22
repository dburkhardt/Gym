# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serial Gym entry point for the unmodified creator Napari/TinyAgent harness.

Inference remains fail-closed until the independently reviewed sandbox/evaluator
packet enables it. Operational solver completion is never a scientific reward.
"""

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

from fastapi import Body, Request
from pydantic import ConfigDict, Field, PrivateAttr

from benchmarks.scivisagentbench.managed import execute
from benchmarks.scivisagentbench.prepare import FROZEN_SHA256
from benchmarks.scivisagentbench.transport import MODEL
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict, get_global_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import current_rollout_id, maybe_rollout_id_from_run_body
from nemo_gym.server_utils import rollout_path_prefix


class Config(BaseResponsesAPIAgentConfig):
    source_root: str
    task_manifest: str
    native_config: str
    output_root: str
    endpoint: str = ""  # Historical metadata only; never used for dispatch.
    native_python: str = ""
    runtime_packet: str = ""
    model_server: ModelServerRef = Field(
        default_factory=lambda: ModelServerRef(type="responses_api_models", name="policy_model")
    )
    max_tokens: int = 16384
    timeout_s: int = 7200
    request_timeout_s: float = Field(default=600, gt=0, allow_inf_nan=False)
    inference_enabled: bool = False
    judge_command: list[str] = Field(default_factory=list)


class RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict


class RunResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    artifact_dir: str
    native_solver_result: dict = Field(default_factory=dict)
    scientific_score: float | None = None
    native_evaluation: dict = Field(default_factory=dict)
    protocol_limitations: list[str] = Field(
        default_factory=lambda: [
            "Creator screenshot formatter converts tool images to text placeholders",
            "Native composite includes operational and efficiency components; not calibrated scientific quality",
        ]
    )


class Agent(SimpleResponsesAPIAgent):
    config: Config
    _semaphore: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(1))

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()):
        raise NotImplementedError("Use /run with an allowlisted frozen case ID")

    async def run(self, body: RunRequest = Body(), request: Request = None) -> RunResponse:
        async with self._semaphore:
            if not self.config.inference_enabled:
                raise ValueError("SciVis inference gate is closed")
            if body.verifier_metadata.get("frozen_manifest_sha256") != FROZEN_SHA256:
                raise ValueError("Request is not bound to frozen eleven-case manifest")
            manifest = Path(self.config.task_manifest).read_bytes()
            if hashlib.sha256(manifest).hexdigest() != FROZEN_SHA256:
                raise ValueError("Frozen eleven-case manifest changed")
            tasks = json.loads(manifest)
            task_id = body.verifier_metadata.get("task_id")
            if task_id not in tasks:
                raise ValueError("unknown frozen task ID")
            task = tasks[task_id]
            global_config = get_global_config_dict()
            if not global_config.get("provider_dispatch_guard") or global_config.get("policy_model_name") != MODEL:
                raise ValueError("Guarded exact-model native service required")
            capture = current_rollout_id() or maybe_rollout_id_from_run_body(body)
            if capture is None:
                raise ValueError("Native rollout identity required")
            attempt_key = hashlib.sha256(json.dumps([task_id, capture]).encode()).hexdigest()
            output = Path(self.config.output_root) / attempt_key
            if output.exists():
                raise ValueError("Existing native attempt requires reconciliation; no implicit retry")
            output.parent.mkdir(parents=True, exist_ok=True)
            # Claim once across native service workers, before any error conversion.
            with output.with_suffix(".claim").open("x") as claim:
                json.dump({"task_id": task_id, "capture_id": capture}, claim)
                claim.flush()
                os.fsync(claim.fileno())
            directory = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)

            server = get_first_server_config_dict(global_config, self.config.model_server.name)
            endpoint = f"http://{server['host']}:{server['port']}" + rollout_path_prefix(capture) + "/v1"
            result = {}
            evaluation = {}
            failure = "native_evaluator_pending"
            try:
                result, evaluation = await execute(self.config, task, output, endpoint)
                if not result.get("success"):
                    failure = "native_solver_failed: " + str(result.get("error"))
                elif evaluation:
                    failure = None if evaluation["valid_for_interpretation"] else evaluation["failure_kind"]
            except Exception as exc:
                failure = type(exc).__name__ + ": " + str(exc)
            response = NeMoGymResponse(
                id=output.name,
                created_at=time.time(),
                model=MODEL,
                object="response",
                output=[],
                parallel_tool_calls=False,
                tools=[],
                tool_choice="auto",
            )
            # Native percentage includes operational components; preserve component breakdown.
            score = (
                evaluation["native_result"]["scores"]["percentage"] / 100
                if evaluation.get("valid_for_interpretation")
                else None
            )
            record = RunResponse(
                **body.model_dump(),
                response=response,
                reward=score if score is not None else 0,
                mask_sample=score is None,
                failure_kind="scivis:unscored" if score is None else None,
                failure_reason=failure,
                scientific_score=score,
                native_solver_result=result,
                native_evaluation=evaluation,
                artifact_dir=str(output),
            )
            output.mkdir(parents=True, exist_ok=True)
            (output / "gym-result.json").write_text(record.model_dump_json(indent=2))
            return record


if __name__ == "__main__":
    Agent.run_webserver()
