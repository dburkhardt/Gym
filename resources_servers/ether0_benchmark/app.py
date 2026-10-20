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
            import json
            import sys
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "resources_servers.ether0_benchmark.score_worker",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            payload = json.dumps({"text": body.response.output_text or "", **body.verifier_metadata})
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(payload.encode()), timeout=120)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise RuntimeError("Creator scorer subprocess exceeded 120 seconds")
            if process.returncode:
                raise RuntimeError(f"Creator scorer subprocess exited {process.returncode}: " + stderr.decode(errors="replace")[-2000:])
            result = json.loads(stdout)
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
