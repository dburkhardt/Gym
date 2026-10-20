# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native MetaSyn BM25/RAG generator and complete upstream evaluator through Gym."""

import asyncio
import json
import math
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from fastapi import Request, Response
from pydantic import PrivateAttr

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming


MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


def native_reward(evaluation: dict) -> float:
    if evaluation.get("status") != "succeeded":
        raise ValueError("Native grading did not succeed; no scientific reward is assigned")
    metric = evaluation.get("metrics", {}).get("inclusion_f1")
    if isinstance(metric, bool) or not isinstance(metric, (int, float)):
        raise ValueError("Missing or nonnumeric native inclusion_f1; no scientific reward is assigned")
    reward = float(metric)
    if not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError("Native inclusion_f1 is nonfinite or out of range")
    return reward


class Config(BaseResponsesAPIAgentConfig):
    native_python: str
    runner_script: Path
    materialized_root: Path
    output_root: Path
    reference_file: Path


class RunRequest(BaseRunRequest):
    verifier_metadata: dict[str, int]


class VerifyResponse(BaseVerifyResponse):
    metrics: dict
    evaluation_status: str
    artifact_dir: str
    reward_metric: str = "inclusion_f1"


class MetaSynAgent(SimpleResponsesAPIAgent):
    config: Config
    _semaphore: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(1))

    async def responses(
        self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        raise NotImplementedError("Use /run with a prepared review identifier")

    def _evaluate(self, review_id: int):
        out = self.config.output_root / str(review_id)
        out.mkdir(parents=True, exist_ok=False)
        with (out / "native-runtime.log").open("x") as log:
            subprocess.run(
                [
                    self.config.native_python,
                    str(self.config.runner_script),
                    "--review-id",
                    str(review_id),
                    "--output-root",
                    str(self.config.output_root),
                    "--reference-file",
                    str(self.config.reference_file),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=3600,
            )
        evaluation = json.loads((out / "evaluation.json").read_text())
        return out, evaluation

    async def run(self, request: Request, body: RunRequest) -> VerifyResponse:
        review_id = body.verifier_metadata["review_id"]
        supplied = body.responses_create_params.model_dump()["input"]
        expected = (self.config.materialized_root / "rag" / str(review_id) / "prompt.txt").read_text()
        if len(supplied) != 1 or supplied[0].get("role") != "user" or supplied[0].get("content") != expected:
            raise ValueError("Gym input differs from the preserved native prompt")
        async with self._semaphore:
            out, evaluation = await asyncio.to_thread(self._evaluate, review_id)
        report = (out / "report.md").read_text()
        response = NeMoGymResponse.model_validate(
            {
                "id": "resp_" + uuid4().hex,
                "object": "response",
                "created_at": time.time(),
                "model": MODEL,
                "status": "completed",
                "output": [
                    {
                        "id": "msg_" + uuid4().hex,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "annotations": [], "text": report}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        # Gym needs one scalar. Preserve the named native component and every other metric;
        # this is not an invented overall benchmark score or a pass/fail claim.
        reward = native_reward(evaluation)
        return VerifyResponse(
            **body.model_dump(),
            response=response,
            reward=reward,
            metrics=evaluation["metrics"],
            evaluation_status=evaluation["status"],
            artifact_dir=str(out),
        )


if __name__ == "__main__":
    MetaSynAgent.run_webserver()
