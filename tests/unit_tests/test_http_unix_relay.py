# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

from nemo_gym.sandbox import http_unix_relay


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/responses"])
async def test_real_relay_exact_route_auth_and_no_retry(route):
    calls = []

    async def upstream(request):
        calls.append(await request.json())
        if request.headers.get("Authorization") != "Bearer fixture":
            raise web.HTTPUnauthorized()
        return web.json_response({"fixture": True})

    with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
        root = Path(temporary)
        app = web.Application()
        app.router.add_post(route, upstream)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.UnixSite(runner, str(root / "s")).start()
        child = subprocess.Popen(
            [
                sys.executable,
                http_unix_relay.__file__,
                "--socket",
                str(root / "s"),
                "--port-file",
                str(root / "ready"),
                "--route",
                route,
                "--max-body-bytes",
                "32",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            async with asyncio.timeout(5):
                while not (root / "ready").exists():
                    assert child.poll() is None
                    await asyncio.sleep(0.01)
            ready = json.loads((root / "ready").read_text())
            assert ready["pid"] == child.pid
            async with aiohttp.ClientSession() as client:
                base = f"http://127.0.0.1:{ready['port']}"
                async with client.post(base + "/arbitrary", json={}) as response:
                    assert response.status == 404
                async with client.post(base + route, data="x" * 33) as response:
                    assert response.status == 413
                assert calls == []
                async with client.post(base + route, json={}) as response:
                    assert response.status == 401
                async with client.post(
                    base + route, json={"test": 1}, headers={"Authorization": "Bearer fixture"}
                ) as response:
                    assert response.status == 200 and await response.json() == {"fixture": True}
            assert calls == [{}, {"test": 1}]
        finally:
            child.terminate()
            await asyncio.to_thread(child.wait, timeout=5)
            await runner.cleanup()


def test_relay_refuses_arbitrary_forwarding(tmp_path):
    with pytest.raises(ValueError, match="protocol routes"):
        http_unix_relay.serve(socket_path="unused", port_file=tmp_path / "unused", route="/admin")
