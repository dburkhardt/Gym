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


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_bridge_rejects_invalid_request_deadline(value):
    with pytest.raises(ValueError, match="finite and positive"):
        scoped_bridge.application(endpoint="http://localhost/ng-rollout/exact/v1", token="t", lineage="l",
                                  digest="d", request_timeout_s=value)


@pytest.mark.asyncio
@pytest.mark.parametrize("deadline,success", [(0.02, False), (0.5, True)])
async def test_configured_deadline_bounds_one_dispatch_without_resend(monkeypatch, deadline, success):
    calls = []
    settled = []

    class Reply:
        status = 200
        async def read(self):
            return b'{"ok":true}'
        def release(self):
            settled.append("released")

    async def send(*args, **kwargs):
        calls.append(kwargs)
        try:
            await asyncio.sleep(0.08)
            return Reply()
        except asyncio.CancelledError:
            settled.append("cancelled")
            raise

    monkeypatch.setattr(scoped_bridge, "request", send)
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="bridge-deadline-") as directory:
        socket = Path(directory) / "model.sock"
        async with scoped_bridge.bridge(socket, endpoint="http://localhost/ng-rollout/exact/v1",
                                        token="t", lineage="l", digest="d", request_timeout_s=deadline):
            async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=str(socket))) as client:
                async with client.post("http://scoped/chat/completions", json={"model": MODEL}, headers={
                    "Authorization": "Bearer t", "x-nemo-gym-logical-call-id": "blade:l:0",
                    "x-nemo-gym-call-config-sha256": "d"}) as reply:
                    assert (reply.status == 200) is success
        assert not socket.exists()
    assert len(calls) == 1
    assert calls[0]["_max_num_tries"] == 1 and not calls[0]["_resend_after_disconnect"]
    assert settled == (["released"] if success else ["cancelled"])
