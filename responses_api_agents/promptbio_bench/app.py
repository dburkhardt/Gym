# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""External Gym agent: Docker solving followed by pinned upstream grading."""

import asyncio
import json
import time
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, Request, Response
from pydantic import Field, PrivateAttr

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.promptbio_bench.runtime import download_task, grade, solve


class PromptBioConfig(BaseResponsesAPIAgentConfig):
    model_name: str
    model_base_url: str
    model_key_file: Path
    sandbox_image: str
    cache_dir: Path
    output_dir: Path
    scorer_dir: Path
    scorer_python: str
    # Pairing is deliberate: both solvers share the upstream Detect/Recommend pass.
    comparison_models: list[str] = Field(default_factory=list)
    judge_model: str = "gpt-5.4"
    judge_key_file: Path
    budget_ledger: Path
    budget_usd: float = 100
    max_steps: int = 100
    max_tokens: int = 16384


class PromptBioRequest(BaseRunRequest):
    verifier_metadata: dict[str, str]


class PromptBioAgent(SimpleResponsesAPIAgent):
    config: PromptBioConfig
    _semaphore: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(1))

    async def responses(
        self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        raise HTTPException(400, "PromptBio requires /run with a task ID")

    def _evaluate(self, task_id: str) -> tuple[float, str]:
        cfg = self.config
        task = download_task(task_id, cfg.cache_dir)
        models = list(dict.fromkeys([cfg.model_name, *cfg.comparison_models]))
        runs = [
            solve(
                task,
                cfg.output_dir,
                model=model,
                base_url=cfg.model_base_url,
                api_key=cfg.model_key_file.read_text().strip(),
                image=cfg.sandbox_image,
                max_steps=cfg.max_steps,
                max_tokens=cfg.max_tokens,
            )
            for model in models
        ]
        scores = grade(
            task,
            runs,
            scorer_dir=cfg.scorer_dir,
            python=cfg.scorer_python,
            output=cfg.output_dir / task_id / "grading",
            judge_model=cfg.judge_model,
            judge_key_file=cfg.judge_key_file,
            budget_ledger=cfg.budget_ledger,
            budget_usd=cfg.budget_usd,
        )
        return scores[0], str(runs[0])

    async def run(self, request: Request, body: PromptBioRequest) -> BaseVerifyResponse:
        async with self._semaphore:
            score, path = await asyncio.to_thread(self._evaluate, body.verifier_metadata["task_id"])
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
                                "text": json.dumps({"artifact_dir": path, "scoring": "upstream avg_similarity"}),
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
