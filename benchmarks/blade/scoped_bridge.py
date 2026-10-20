# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-scoped UDS model forwarding; the scientific container has no IP network."""

import asyncio
import hmac
from contextlib import asynccontextmanager
from pathlib import Path

from aiohttp import web

from nemo_gym.server_utils import request
from responses_api_agents.harbor_agent.custom_verifiers.rewardkit_agent.verifier import _settle


MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"
MAX_BODY = 16 * 1024 * 1024


def application(*, endpoint: str, token: str, lineage: str, digest: str) -> web.Application:
    """Expose only this trial's exact model endpoint and dispatch identities."""
    if "/ng-rollout/" not in endpoint or not endpoint.endswith("/v1"):
        raise ValueError("Bridge target must be a guarded Gym capture endpoint")
    app = web.Application(client_max_size=MAX_BODY)

    async def forward(incoming: web.Request) -> web.Response:
        if not hmac.compare_digest(incoming.headers.get("Authorization", ""), "Bearer " + token):
            raise web.HTTPUnauthorized()
        call_id = incoming.headers.get("x-nemo-gym-logical-call-id", "")
        ordinal = call_id.removeprefix(f"blade:{lineage}:")
        if (
            not call_id.startswith(f"blade:{lineage}:")
            or not ordinal.isdigit()
            or incoming.headers.get("x-nemo-gym-call-config-sha256") != digest
        ):
            raise web.HTTPForbidden()
        payload = await incoming.json()
        if not isinstance(payload, dict) or payload.get("model") != MODEL or payload.get("stream"):
            raise web.HTTPBadRequest()
        headers = {"x-nemo-gym-logical-call-id": call_id, "x-nemo-gym-call-config-sha256": digest}
        async with asyncio.timeout(600):
            response = await request(
                "POST",
                endpoint + "/chat/completions",
                json=payload,
                headers=headers,
                allow_redirects=False,
                _max_num_tries=1,
                _max_connection_retries=0,
                _resend_after_disconnect=False,
            )
            try:
                return web.Response(
                    status=response.status, body=await response.read(), content_type="application/json"
                )
            finally:
                response.release()

    app.router.add_post("/chat/completions", forward)
    return app


@asynccontextmanager
async def bridge(socket_path: Path, *, endpoint: str, token: str, lineage: str, digest: str):
    """Clean up only this owned socket after active forwarding handlers settle."""
    runner = web.AppRunner(
        application(endpoint=endpoint, token=token, lineage=lineage, digest=digest), shutdown_timeout=610
    )
    await runner.setup()
    try:
        await web.UnixSite(runner, str(socket_path)).start()
        socket_path.chmod(0o600)
        yield
    finally:
        try:
            await _settle(asyncio.create_task(runner.cleanup()))
        finally:
            socket_path.unlink(missing_ok=True)
