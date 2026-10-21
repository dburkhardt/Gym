# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Gym verifier; references never enter model messages."""

from hashlib import sha256
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict, Field

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
from resources_servers.ctbench.grading import PIN, creator, parse_and_score


class Config(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef | None = None
    creator_root: str
    judge_max_output_tokens: int = Field(default=1000, gt=0)
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class Request(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
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
            max_output_tokens=self.config.judge_max_output_tokens,
        )
        judged = await call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=params,
            response_model=NeMoGymResponse,
            verification_id=body.verification_id,
            call_id="feature-matching",
            scorer_identity={
                "protocol": "ctbench-zero-shot-nemotron-judge-v1",
                "creator_revision": PIN,
                "creator_source_sha256": functions["__creator_source_sha256"],
                "scorer_sha256": sha256(Path(__file__).with_name("grading.py").read_bytes()).hexdigest(),
                "verifier_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "judge_model": params.model,
                "temperature": params.temperature,
                "max_output_tokens": params.max_output_tokens,
            },
        )
        if judged.model != "nvidia/nvidia/nemotron-3.5-super-vl":
            raise RuntimeError("Judge model is outside benchmark allowlist")
        if judged.status != "completed":
            return self.judge_failure(body, judged, f"Judge status {judged.status}: {judged.incomplete_details}")
        if not (judged.output_text or "").strip():
            return self.judge_failure(body, judged, "Judge returned empty text")
        try:
            scores = parse_and_score(judged.output_text or "", references, candidates, functions)
        except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            return self.judge_failure(body, judged, f"Creator scorer or judge structure failure: {exc}")
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
