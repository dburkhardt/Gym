# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Gym verifier; references never enter model messages."""

from typing import ClassVar

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.completion_outcome import with_completion_outcome
from resources_servers.superchem.grading import grade


class Config(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class Request(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict


class Response(BaseVerifyResponse, Request):
    diagnostics: dict


class Verifier:
    async def verify(self, body: Request) -> Response:
        import asyncio

        result = await asyncio.to_thread(grade, body.response.output_text or "", **body.verifier_metadata)
        return Response(**with_completion_outcome({**body.model_dump(), **result}, response=body.response))


class Server(Verifier, SimpleResourcesServer):
    config: Config
    ray_enabled = False


if __name__ == "__main__":
    Server.run_webserver()
