# SPDX-License-Identifier: Apache-2.0
import asyncio
import tempfile
from pathlib import Path

import aiohttp
import pytest

from nemo_gym.sandbox import model_bridge


@pytest.mark.asyncio
async def test_sse_is_forwarded_before_upstream_completion(monkeypatch):
    proceed = asyncio.Event()
    released = []

    class Content:
        async def iter_chunked(self, size):
            yield b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
            await proceed.wait()
            yield b"data: [DONE]\n\n"

    class Reply:
        status = 200
        headers = {"Content-Type": "text/event-stream"}
        content = Content()

        def release(self):
            released.append(True)

    async def send(*args, **kwargs):
        assert kwargs["json"]["stream"] is True
        assert kwargs["_max_num_tries"] == 1
        return Reply()

    monkeypatch.setattr(model_bridge, "request", send)
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        path = Path(directory) / "model.sock"
        async with model_bridge.bridge(
            path,
            endpoint="http://native/ng-rollout/task/v1",
            token="secret",
            lineage="lineage",
            digest="digest",
            namespace="corral",
            allow_stream=True,
        ):
            async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=str(path))) as client:
                async with client.post(
                    "http://local/chat/completions",
                    json={"model": model_bridge.MODEL, "stream": True},
                    headers={
                        "Authorization": "Bearer secret",
                        "x-nemo-gym-logical-call-id": "corral:lineage:0",
                        "x-nemo-gym-call-config-sha256": "digest",
                    },
                ) as result:
                    first = await asyncio.wait_for(result.content.readline(), 2)
                    assert b"first" in first
                    assert not proceed.is_set()
                    proceed.set()
                    assert b"[DONE]" in await result.read()
        assert released == [True]
        assert not path.exists()


@pytest.mark.asyncio
async def test_stream_body_timeout_releases_and_removes_socket(monkeypatch):
    released = asyncio.Event()

    class Content:
        async def iter_chunked(self, size):
            yield b'data: {"first":true}\n\n'
            await asyncio.Event().wait()

    class Reply:
        status = 200
        headers = {"Content-Type": "text/event-stream"}
        content = Content()

        def release(self):
            released.set()

    async def send(*args, **kwargs):
        return Reply()

    monkeypatch.setattr(model_bridge, "request", send)
    monkeypatch.setattr(model_bridge, "REQUEST_TIMEOUT_S", 0.05)
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        path = Path(directory) / "model.sock"
        async with model_bridge.bridge(
            path,
            endpoint="http://native/ng-rollout/task/v1",
            token="secret",
            lineage="lineage",
            digest="digest",
            namespace="corral",
            allow_stream=True,
        ):
            async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=str(path))) as client:
                async with client.post(
                    "http://local/chat/completions",
                    json={"model": model_bridge.MODEL, "stream": True},
                    headers={
                        "Authorization": "Bearer secret",
                        "x-nemo-gym-logical-call-id": "corral:lineage:0",
                        "x-nemo-gym-call-config-sha256": "digest",
                    },
                ) as result:
                    with pytest.raises(aiohttp.ClientPayloadError):
                        await asyncio.wait_for(result.read(), 2)
            assert released.is_set()
        assert not path.exists()
