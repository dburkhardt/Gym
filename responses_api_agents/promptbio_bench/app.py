# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""External Gym agent: Docker solving followed by pinned upstream grading."""

import asyncio
import json
import time
from pathlib import Path
from uuid import uuid4

from fastapi import Request, Response
from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import current_rollout_id, maybe_rollout_id_from_run_body
from nemo_gym.server_utils import rollout_path_prefix
from responses_api_agents.promptbio_bench.managed_runtime import (
    download_managed,
    grade_managed,
    solve_managed,
    validate_image,
)


class PromptBioConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    model_name: str
    sandbox_image: str
    cache_dir: Path
    output_dir: Path
    scorer_dir: Path
    scorer_python: str
    scorer_image: str | None = None
    # Pairing is deliberate: both solvers share the upstream Detect/Recommend pass.
    comparison_models: list[str] = Field(default_factory=list)
    judge_model: str = "nvidia/nvidia/nemotron-3.5-super-vl"
    max_steps: int = 100
    max_tokens: int = 16384

    @model_validator(mode="after")
    def campaign_contract(self):
        from responses_api_agents.promptbio_bench.nemotron_proxy import MODEL

        if self.judge_model != MODEL or any(model != MODEL for model in [self.model_name, *self.comparison_models]):
            raise ValueError("Campaign requires Super for every role")
        return self


class PromptBioRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: dict[str, str]


class PromptBioAgent(SimpleResponsesAPIAgent):
    config: PromptBioConfig
    _semaphore: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(1))

    async def responses(
        self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        raise NotImplementedError("PromptBio requires /run with a task ID")

    async def _evaluate(self, task_id: str, endpoint: str, rollout_id: str) -> tuple[float, str]:
        cfg = self.config
        validate_image(cfg.scorer_image)
        validate_image(cfg.sandbox_image)
        task = await download_managed(task_id, cfg.cache_dir, cfg.output_dir / "acquisition" / (uuid4().hex + ".json"))
        models = list(dict.fromkeys([cfg.model_name, *cfg.comparison_models]))
        runs = []
        for model in models:
            runs.append(
                await solve_managed(
                    task,
                    cfg.output_dir,
                    endpoint=endpoint,
                    rollout_id=rollout_id,
                    model=model,
                    image=cfg.sandbox_image,
                    max_steps=cfg.max_steps,
                    max_tokens=cfg.max_tokens,
                )
            )
        scores = await grade_managed(
            task,
            runs,
            scorer_dir=cfg.scorer_dir,
            python=cfg.scorer_python,
            output=cfg.output_dir / task_id / ("grading-" + runs[0].name),
            endpoint=endpoint,
            rollout_id=rollout_id,
            scorer_image=cfg.scorer_image,
        )
        return scores[0], str(runs[0])

    async def run(self, request: Request, body: PromptBioRequest) -> BaseVerifyResponse:
        if self.config.model_server is None:
            raise ValueError("PromptBio native execution requires a guarded Gym model_server reference")
        global_config = self.server_client.global_config_dict
        if not global_config.get("provider_dispatch_guard"):
            raise ValueError("PromptBio native execution requires provider_dispatch_guard")
        rollout_id = current_rollout_id() or maybe_rollout_id_from_run_body(body)
        if not rollout_id:
            raise ValueError("PromptBio requires a native rollout identity")
        server = get_first_server_config_dict(global_config, self.config.model_server.name)
        endpoint = f"http://{server['host']}:{server['port']}" + rollout_path_prefix(rollout_id) + "/v1"
        async with self._semaphore:
            score, path = await self._evaluate(body.verifier_metadata["task_id"], endpoint, rollout_id)
        response = NeMoGymResponse.model_validate(
            {
                "id": "resp_" + uuid4().hex,
                "object": "response",
                "created_at": time.time(),
                "model": self.config.model_name,
                "status": "completed",
                "output": [
                    {
                        "id": "msg_" + uuid4().hex,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "annotations": [],
                                "text": json.dumps(
                                    {
                                        "artifact_dir": path,
                                        "scoring": "upstream avg_similarity",
                                        "solver_disposition": json.loads((Path(path) / "receipt.json").read_text())[
                                            "status"
                                        ],
                                    }
                                ),
                            }
                        ],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        return BaseVerifyResponse(**body.model_dump(), response=response, reward=score)


if __name__ == "__main__":
    PromptBioAgent.run_webserver()
