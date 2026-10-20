# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Gym verifier; references never enter model messages."""

from typing import ClassVar

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import call_judge
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from resources_servers.ctbench.grading import creator, parse_and_score


class Config(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef | None = None
    creator_root: str
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class Request(BaseVerifyRequest):
    verifier_metadata: dict


class Response(BaseVerifyResponse, Request):
    diagnostics: dict


class Verifier:
    async def verify(self, body: Request) -> Response:
        if not self.config.judge_model_server:
            raise RuntimeError("CTBench requires Nemotron judge")
        meta = body.verifier_metadata
        functions = creator(self.config.creator_root)
        references = functions["extract_elements_v2"](meta["reference"])
        candidates = functions["extract_elements_v2"](body.response.output_text or "")
        system, query = functions["build_gpt4_eval_prompt"](references, candidates, meta["trial_context"])
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            model="nvidia/nvidia/nemotron-3.5-super-vl",
            temperature=0,
            max_output_tokens=1000,
        )
        judged = await call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=params,
            response_model=NeMoGymResponse,
        )
        if judged.model != "nvidia/nvidia/nemotron-3.5-super-vl":
            raise RuntimeError("Judge model is outside benchmark allowlist")
        if judged.status != "completed":
            raise RuntimeError("Judge did not complete; preserve as grader failure")
        scores = parse_and_score(judged.output_text or "", references, candidates, functions)
        return Response(
            **body.model_dump(),
            reward=float(scores["f1"]),
            diagnostics={
                "protocol": "CTBench zero-shot Nemotron judge variant",
                "scores": scores,
                "judge_response": judged.model_dump(),
                "deviation": "Nemotron replaces GPT-4o judge; feature accounting integrity guard added",
            },
        )


class Server(Verifier, SimpleResourcesServer):
    config: Config
    ray_enabled = False


if __name__ == "__main__":
    Server.run_webserver()
