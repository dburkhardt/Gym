# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Gym verifier; references never enter model messages."""

from pathlib import Path
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
from resources_servers.biomaze.grading import grade
from resources_servers.biomaze.openended import judge_query, parse_judge


class Config(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef | None = None
    judge_prompt_file: str | None = None
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class Request(BaseVerifyRequest):
    verifier_metadata: dict


class Response(BaseVerifyResponse, Request):
    diagnostics: dict


class Verifier:
    def judge_failure(self, body: Request, judged: NeMoGymResponse, reason: str) -> Response:
        return Response(
            **body.model_dump(),
            reward=0.0,
            mask_sample=True,
            failure_kind="judge_failed",
            failure_reason=reason,
            diagnostics={
                "judge_response": judged.model_dump(),
                "failure_reason": reason,
                "reward_is_serialization_placeholder": True,
                "denominator_policy": "Retain attempted item; report judge failure separately from performance",
            },
        )

    async def verify(self, body: Request) -> Response:
        import asyncio

        if body.verifier_metadata.get("protocol") == "openended":
            if not self.config.judge_model_server or not self.config.judge_prompt_file:
                raise RuntimeError("Openended requires Nemotron judge and pinned creator judge prompt")
            meta = body.verifier_metadata
            query = judge_query(
                Path(self.config.judge_prompt_file).read_text(),
                meta["question"],
                meta["answer"],
                body.response.output_text or "",
            )
            params = NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": query}],
                model="nvidia/nvidia/nemotron-3.5-super-vl",
                temperature=0,
                max_output_tokens=4096,
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
                return self.judge_failure(body, judged, f"Judge status {judged.status}: {judged.incomplete_details}")
            if not (judged.output_text or "").strip():
                return self.judge_failure(body, judged, "Judge returned empty text")
            return Response(
                **body.model_dump(),
                reward=float(parse_judge(judged.output_text or "")),
                diagnostics={
                    "protocol": "Nemotron conclusion-judge variant",
                    "judge_response": judged.model_dump(),
                    "deviation": "Creator 3800-token model-specific input gate replaced by provider context enforcement",
                },
            )
        result = await asyncio.to_thread(grade, body.response.output_text or "", **body.verifier_metadata)
        return Response(**body.model_dump(), **result)


class Server(Verifier, SimpleResourcesServer):
    config: Config
    ray_enabled = False


if __name__ == "__main__":
    Server.run_webserver()
