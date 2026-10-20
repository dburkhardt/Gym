# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import tempfile
from pathlib import Path

import aiohttp
import pytest

from benchmarks.blade.native_phase import MODEL, post_json
from nemo_gym.sandbox import model_bridge as scoped_bridge


@pytest.mark.asyncio
async def test_real_unix_roundtrip_and_scope_rejections(monkeypatch):
    calls = []

    class Reply:
        status = 200

        async def read(self):
            return b'{"ok":true}'

        def release(self):
            pass

    async def send(method, url, **kwargs):
        calls.append((url, kwargs))
        assert not kwargs["allow_redirects"] and kwargs["_max_num_tries"] == 1
        assert "Authorization" not in kwargs["headers"]
        return Reply()

    monkeypatch.setattr(scoped_bridge, "request", send)
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="blade-test-") as directory:
        socket = Path(directory) / "model.sock"
        headers = {
            "Authorization": "Bearer secret",
            "x-nemo-gym-logical-call-id": "blade:lineage:0",
            "x-nemo-gym-call-config-sha256": "digest",
        }
        async with scoped_bridge.bridge(
            socket, endpoint="http://localhost/ng-rollout/exact/v1", token="secret", lineage="lineage", digest="digest"
        ):
            assert await asyncio.to_thread(post_json, str(socket), {"model": MODEL}, headers) == {"ok": True}
            async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=str(socket))) as client:
                for path, payload, overrides, status in [
                    ("/anything", {"model": MODEL}, {}, 404),
                    ("/chat/completions", {"model": MODEL}, {"Authorization": "Bearer wrong"}, 401),
                    ("/chat/completions", {"model": "other"}, {}, 400),
                    ("/chat/completions", {"model": MODEL}, {"x-nemo-gym-logical-call-id": "blade:another:0"}, 403),
                    ("/chat/completions", {"model": MODEL}, {"x-nemo-gym-call-config-sha256": "other"}, 403),
                ]:
                    async with client.post(
                        "http://scoped" + path, json=payload, headers={**headers, **overrides}
                    ) as result:
                        assert result.status == status
                async with client.post(
                    "http://scoped/chat/completions", data=b"x" * (scoped_bridge.MAX_BODY + 1), headers=headers
                ) as result:
                    assert result.status == 413
            assert len(calls) == 1
        assert not socket.exists()


@pytest.mark.asyncio
async def test_cancellation_waits_for_active_forwarder_and_removes_socket(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()

    class Reply:
        status = 200

        async def read(self):
            await release.wait()
            return b"{}"

        def release(self):
            pass

    async def send(*args, **kwargs):
        entered.set()
        return Reply()

    monkeypatch.setattr(scoped_bridge, "request", send)
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="blade-test-") as directory:
        socket = Path(directory) / "model.sock"

        async def serve():
            async with scoped_bridge.bridge(
                socket,
                endpoint="http://localhost/ng-rollout/exact/v1",
                token="secret",
                lineage="lineage",
                digest="digest",
            ):
                await asyncio.Event().wait()

        task = asyncio.create_task(serve())
        while not socket.exists():
            await asyncio.sleep(0.001)
        call = asyncio.create_task(
            asyncio.to_thread(
                post_json,
                str(socket),
                {"model": MODEL},
                {
                    "Authorization": "Bearer secret",
                    "x-nemo-gym-logical-call-id": "blade:lineage:0",
                    "x-nemo-gym-call-config-sha256": "digest",
                },
            )
        )
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        release.set()
        await call
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not socket.exists()
