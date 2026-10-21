# SPDX-License-Identifier: Apache-2.0
"""Maintained creator FlyAOC workflow using managed solver/scorer phases."""

import asyncio
import hashlib
import json
import math
import time
from pathlib import Path
from uuid import uuid4

from pydantic import ConfigDict, PrivateAttr

from benchmarks.flyaoc.managed import MODEL, run_scorer, run_solver
from benchmarks.flyaoc.trajectory import build_trajectory, summed_usage
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict, get_global_config_dict
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_correlation import current_rollout_id, maybe_rollout_id_from_run_body
from nemo_gym.server_utils import rollout_path_prefix


class Config(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    runtime_image: str
    solver_assets: Path
    references: Path
    output_root: Path


class RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: dict


class VerifyResponse(BaseVerifyResponse):
    metrics: dict
    ng_trajectory: dict
    artifact_dir: str
    reward_metric: str = "task1_semantic_recall_at_30_micro"


class FlyAOCAgent(SimpleResponsesAPIAgent):
    config: Config
    _semaphore: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(1))

    async def responses(self, *args, **kwargs):
        raise NotImplementedError("Use the prepared native /run workflow")

    async def run(self, body: RunRequest) -> VerifyResponse:
        global_config = get_global_config_dict()
        rollout = current_rollout_id() or maybe_rollout_id_from_run_body(body)
        if not global_config.get("provider_dispatch_guard") or not rollout:
            raise ValueError("Native FlyAOC requires guarded rollout identity")
        if global_config["policy_model_name"] != MODEL:
            raise ValueError("Unexpected campaign model")
        messages = body.responses_create_params.model_dump()["input"]
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise ValueError("Expected one frozen public gene record")
        payload = json.loads(messages[0]["content"])
        if payload["gene_id"] != body.verifier_metadata["gene_id"]:
            raise ValueError("Input gene identity mismatch")
        server = get_first_server_config_dict(global_config, self.config.model_server.name)
        endpoint = f"http://{server['host']}:{server['port']}" + rollout_path_prefix(rollout) + "/v1"
        trial = self.config.output_root / hashlib.sha256(str(rollout).encode()).hexdigest()
        async with self._semaphore:
            predictions = await run_solver(
                image=self.config.runtime_image,
                root=self.config.solver_assets,
                output=trial / "solver",
                payload=payload,
                endpoint=endpoint,
            )
            graded = await run_scorer(
                image=self.config.runtime_image,
                root=self.config.solver_assets,
                output=trial / "scorer",
                predictions=predictions,
                references=self.config.references,
            )
        metrics = graded["evaluation"]["aggregate"]
        reward = metrics["task1_semantic_recall_at_30_micro"]
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(reward)
            or not 0 <= reward <= 1
        ):
            raise ValueError("Missing or invalid creator scalar metric")
        response = NeMoGymResponse.model_validate(
            {
                "id": "resp_" + uuid4().hex,
                "object": "response",
                "created_at": time.time(),
                "model": MODEL,
                "status": "completed",
                "usage": summed_usage(trial / "solver" / "model-calls"),
                "output": [
                    {
                        "id": "msg_" + uuid4().hex,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "annotations": [], "text": predictions.read_text()}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        return VerifyResponse(
            **body.model_dump(),
            response=response,
            reward=reward,
            metrics=metrics,
            artifact_dir=str(trial),
            ng_trajectory=build_trajectory(
                trial / "solver" / "model-calls",
                task_id=payload["gene_id"],
                rollout_id=rollout,
                model_ref=self.config.model_server,
            ).model_dump(mode="json"),
        )


if __name__ == "__main__":
    FlyAOCAgent.run_webserver()
