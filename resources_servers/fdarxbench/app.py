# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Gym verifier; references never enter model messages."""

from hashlib import sha256
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict

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
from resources_servers.fdarxbench.grading import judge_query, parse_judge


class Config(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef | None = None
    judge_prompt_file: str | None = None
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
        if not self.config.judge_model_server or not self.config.judge_prompt_file:
            raise RuntimeError("FDARx requires Nemotron judge and pinned creator judge prompt")
        meta = body.verifier_metadata
        prompt_bytes = Path(self.config.judge_prompt_file).read_bytes()
        query = judge_query(
            prompt_bytes.decode("utf-8"),
            meta["question"],
            meta["answer"],
            body.response.output_text or "",
        )
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"role": "system", "content": "You are a grading assistant."},
                {"role": "user", "content": query},
            ],
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
            verification_id=body.verification_id,
            call_id="fda-rubric",
            scorer_identity={
                "protocol": "fdarx-nemotron-judge-v1",
                "prompt_sha256": sha256(prompt_bytes).hexdigest(),
                "scorer_sha256": sha256(Path(__file__).with_name("grading.py").read_bytes()).hexdigest(),
                "verifier_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "model": params.model,
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
            label = parse_judge(judged.output_text or "")
        except ValueError as exc:
            return self.judge_failure(body, judged, str(exc))
        return Response(
            **body.model_dump(),
            reward=float(label == "A"),
            diagnostics={
                "protocol": "FDARx Nemotron judge variant",
                "judge_label": label,
                "judge_response": judged.model_dump(),
                "deviation": "Nemotron judge replaces reported judge; exact creator rubric retained",
            },
        )


class Server(Verifier, SimpleResourcesServer):
    config: Config
    ray_enabled = False


if __name__ == "__main__":
    Server.run_webserver()
