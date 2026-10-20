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
from resources_servers.ether0_benchmark.grading import grade


class Config(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class Request(BaseVerifyRequest):
    verifier_metadata: dict


class Response(BaseVerifyResponse, Request):
    diagnostics: dict


class Verifier:
    async def verify(self, body: Request) -> Response:
        import asyncio

        try:
            result = await asyncio.to_thread(grade, body.response.output_text or "", **body.verifier_metadata)
        except RuntimeError as exc:
            return Response(
                **body.model_dump(),
                reward=0.0,
                mask_sample=True,
                failure_kind="ether0_benchmark:grader_unavailable",
                failure_reason=str(exc),
                diagnostics={
                    "reward_is_serialization_placeholder": True,
                    "denominator_policy": "Retain attempt; exclude placeholder from scientific performance.",
                    "oracle_error": str(exc),
                },
            )
        return Response(**body.model_dump(), **result)


class Server(Verifier, SimpleResourcesServer):
    config: Config
    ray_enabled = False


if __name__ == "__main__":
    Server.run_webserver()
