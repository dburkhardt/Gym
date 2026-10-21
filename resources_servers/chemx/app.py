# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native deterministic ChemX column-multiset verification; no model judge."""

import asyncio
import hashlib
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict

from benchmarks.chemx.protocol import MANIFEST, ModelOutputError, score
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)


class Config(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    upstream_repo: str


class Request(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    uuid: str
    verifier_metadata: dict


class Response(BaseVerifyResponse, Request):
    scoring_details: dict


class Server(SimpleResourcesServer):
    config: Config
    ray_enabled = False

    async def verify(self, body: Request) -> Response:
        if body.verifier_metadata.get("manifest_sha256") != hashlib.sha256(MANIFEST.read_bytes()).hexdigest():
            raise ValueError("ChemX frozen manifest identity mismatch")
        if body.uuid != body.verifier_metadata.get("article_id"):
            raise ValueError("ChemX article identity mismatch")
        payload = body.model_dump(by_alias=True)
        try:
            result = await asyncio.to_thread(
                score, Path(self.config.upstream_repo), article_id=body.uuid, answer=body.response.output_text or ""
            )
        except ModelOutputError as exc:
            # Creator parsing supplies no score. Preserve the attempted model failure
            # separately from source/scorer exceptions; zero is only a masked placeholder.
            budget = (
                body.response.status == "incomplete"
                and getattr(body.response.incomplete_details, "reason", None) == "max_output_tokens"
            )
            result = {
                "raw_reward": None,
                "failure_origin": "model_output",
                "model_outcome": "output_budget_exhausted" if budget else "invalid_json",
                "reason": str(exc),
            }
            payload.update(reward=0.0, mask_sample=True, scoring_details=result)
            payload["failure_kind"] = payload.get("failure_kind") or (
                "chemx:model_budget" if budget else "chemx:invalid_model_output"
            )
            payload["failure_reason"] = payload.get(
                "failure_reason"
            ) or "Creator parser produced no scientific score: " + str(exc)
        else:
            payload.update(reward=result["reward"], scoring_details=result)
            # Regrading does not erase a transport/model failure or its existing mask.
            payload.setdefault("mask_sample", False)
            payload.setdefault("failure_kind", None)
            payload.setdefault("failure_reason", None)
            if body.response.status == "incomplete" and not payload.get("failure_kind"):
                payload["failure_kind"] = (
                    "chemx:model_budget"
                    if getattr(body.response.incomplete_details, "reason", None) == "max_output_tokens"
                    else "chemx:incomplete_model_output"
                )
                payload["failure_reason"] = "Model response incomplete; parsed creator score retained"
        return Response(**payload)


if __name__ == "__main__":
    Server.run_webserver()
