# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Gym verifier; references never enter model messages."""

from typing import ClassVar, Literal

from fastapi import HTTPException
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from resources_servers.csmed_ft.grading import grade


class Config(BaseResourcesServerConfig):
    scoring_protocol: Literal["whole-paper-strict", "creator-chunk-labels"] = "whole-paper-strict"
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class SeedRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict


class Request(BaseVerifyRequest):
    verifier_metadata: dict


class Response(BaseVerifyResponse, Request):
    diagnostics: dict


class Verifier:
    async def verify(self, body: Request) -> Response:
        import asyncio

        if getattr(getattr(self, "config", None), "scoring_protocol", "whole-paper-strict") == "creator-chunk-labels":
            from benchmarks.csmed_ft.creator_protocol import PROTOCOL, parse_label

            if body.verifier_metadata.get("protocol") != PROTOCOL:
                raise HTTPException(status_code=422, detail="Creator chunk protocol binding required")
            prediction = parse_label(body.response.output_text or "")
            result = {
                "reward": float(prediction == int(body.verifier_metadata["answer"] == "included")),
                "diagnostics": {
                    "prediction": prediction,
                    "protocol": PROTOCOL,
                    "metric_scope": "chunk_only_not_paper_accuracy",
                },
            }
        else:
            result = await asyncio.to_thread(grade, body.response.output_text or "", **body.verifier_metadata)
        return Response(**body.model_dump(), **result)


class Server(Verifier, SimpleResourcesServer):
    config: Config
    ray_enabled = False

    async def seed_session(
        self, body: ResourcesSeedSessionRequest | SeedRequest
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        payload = body.task_data if isinstance(body, ResourcesSeedSessionRequest) else body.model_dump()
        meta = payload.get("verifier_metadata", {})
        if not isinstance(meta, dict):
            meta = {}
        if self.config.scoring_protocol == "creator-chunk-labels":
            from benchmarks.csmed_ft.creator_protocol import PROTOCOL

            if meta.get("protocol") != PROTOCOL or not meta.get("recipe_sha256"):
                raise HTTPException(
                    status_code=422,
                    detail={
                        "failure_kind": "csmed_ft:protocol_mismatch",
                        "task_id": meta.get("task_id"),
                        "dispatch_status": "not_sent",
                    },
                )
        # Frozen preparation stores validity rather than separate raw text fields.
        # Do not reconstruct criteria from the rendered prompt or invent replacements.
        if meta.get("input_valid") is not True:
            raise HTTPException(
                status_code=422,
                detail={
                    "failure_kind": "csmed_ft:invalid_input",
                    "reason": "Missing or unverified full text/eligibility criteria in frozen source",
                    "task_id": meta.get(
                        "task_id",
                        body.task_id.task_id
                        if isinstance(body, ResourcesSeedSessionRequest)
                        else payload.get("task_id"),
                    ),
                    "review_id": meta.get("review_id"),
                    "dispatch_status": "not_sent",
                },
            )
        return await super().seed_session(body)


if __name__ == "__main__":
    Server.run_webserver()
