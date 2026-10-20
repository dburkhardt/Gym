# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Gym verifier; references never enter model messages."""

import asyncio
import math
from typing import ClassVar

from aiohttp import ClientError, ClientTimeout

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import request


class Config(BaseResourcesServerConfig):
    creator_url: str
    creator_timeout_seconds: float = 300.0
    docking_oracle: str = "pyscreener-vina"
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class Request(BaseVerifyRequest):
    verifier_metadata: dict


class Response(BaseVerifyResponse, Request):
    diagnostics: dict


class Verifier:
    async def verify(self, body: Request) -> Response:
        payload = {"query": [body.response.output_text or ""], "metadata": [body.verifier_metadata["metadata"]]}
        try:
            # Docking is expensive and may continue after a dropped connection.
            # One physical request; reconcile the creator before any new attempt.
            received = await request(
                "POST",
                self.config.creator_url.rstrip("/") + "/get_reward",
                json=payload,
                timeout=ClientTimeout(total=self.config.creator_timeout_seconds),
                _max_num_tries=1,
                _resend_after_disconnect=False,
            )
            async with received:
                received.raise_for_status()
                receipt = await received.json()
                if not isinstance(receipt, dict):
                    raise ValueError("Creator response must be an object")
        except (ClientError, asyncio.TimeoutError, ValueError) as exc:
            receipt = {"error": repr(exc), "request": payload}
        reward = receipt.get("reward")
        error = receipt.get("error")
        if error or isinstance(reward, bool) or not isinstance(reward, (float, int)) or not math.isfinite(reward):
            return Response(
                **body.model_dump(),
                reward=0.0,
                mask_sample=True,
                failure_kind="verifier_error",
                failure_reason=str(error or "Invalid creator reward"),
                diagnostics={"creator_response": receipt, "reward_is_serialization_placeholder": True},
            )
        return Response(
            **body.model_dump(),
            reward=float(reward),
            diagnostics={
                "creator_response": receipt,
                "docking_oracle": self.config.docking_oracle,
                "scope": "one-completion reward; diversity-aware top-k aggregate pending",
            },
        )


class Server(Verifier, SimpleResourcesServer):
    config: Config
    ray_enabled = False


if __name__ == "__main__":
    Server.run_webserver()
