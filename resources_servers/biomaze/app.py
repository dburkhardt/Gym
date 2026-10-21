# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Gym verifier; references never enter model messages."""

from hashlib import sha256
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
from resources_servers.biomaze.openended import judge_format_valid, judge_query, parse_judge


class Config(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef | None = None
    judge_prompt_file: str | None = None
    judge_model: str = "nvidia/nvidia/nemotron-3.5-super-vl"
    allowed_judge_models: list[str] = ["nvidia/nvidia/nemotron-3.5-super-vl"]
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
                raise RuntimeError("Openended requires a configured judge and pinned creator judge prompt")
            if self.config.judge_model not in self.config.allowed_judge_models:
                raise RuntimeError("Configured judge model is outside campaign allowlist")
            meta = body.verifier_metadata
            query = judge_query(
                Path(self.config.judge_prompt_file).read_text(),
                meta["question"],
                meta["answer"],
                body.response.output_text or "",
            )
            params = NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": query}],
                model=self.config.judge_model,
                temperature=0,
                max_output_tokens=4096,
            )
            judged = await call_judge(
                self.server_client,
                server_name=self.config.judge_model_server.name,
                url_path="/v1/responses",
                json=params,
                response_model=NeMoGymResponse,
                verification_id=body.verification_id,
                call_id="conclusion",
                scorer_identity={
                    "protocol": "biomaze-conclusion-judge-v1",
                    "prompt_sha256": sha256(Path(self.config.judge_prompt_file).read_bytes()).hexdigest(),
                    "scorer_sha256": sha256(Path(__file__).with_name("openended.py").read_bytes()).hexdigest(),
                    "verifier_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                },
            )
            if judged.model not in self.config.allowed_judge_models or judged.model != self.config.judge_model:
                raise RuntimeError("Judge model is outside benchmark allowlist")
            if judged.status != "completed":
                return self.judge_failure(body, judged, f"Judge status {judged.status}: {judged.incomplete_details}")
            if not (judged.output_text or "").strip():
                return self.judge_failure(body, judged, "Judge returned empty text")
            raw_reward = float(parse_judge(judged.output_text or ""))
            valid = judge_format_valid(judged.output_text or "")
            return Response(
                **body.model_dump(),
                reward=raw_reward,
                mask_sample=not valid,
                failure_kind=None if valid else "judge_unparseable",
                failure_reason=None if valid else "Judge did not return one explicit Correct/Incorrect verdict",
                diagnostics={
                    "protocol": "Configured conclusion-judge variant",
                    "creator_raw_reward": raw_reward,
                    "judge_format_valid": valid,
                    "validity_policy": "explicit-creator-verdict-v1",
                    "denominator_policy": "Retain attempted item; malformed judge excluded from scientific score",
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
