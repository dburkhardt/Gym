# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json

import httpx
import pytest

from benchmarks.bioagentbench import transport


FIRST = b'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"model","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n'


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", [b"data: [DONE]\n\n", b"", asyncio.CancelledError()])
async def test_native_stream_durability_and_poison(tmp_path, monkeypatch, ending):
    calls = []

    class Response:
        status = 200
        headers = {"content-type": "text/event-stream"}
        released = False

        @property
        def content(self):
            return self

        async def iter_chunked(self, size):
            yield FIRST
            if isinstance(ending, BaseException):
                raise ending
            yield ending

        def release(self):
            self.released = True

    raw = Response()

    async def send(*args, **kwargs):
        calls.append(kwargs)
        return raw

    monkeypatch.setattr(transport, "request", send)
    adapter = transport.NativeTransport(
        endpoint="http://local/ng-rollout/one/v1",
        lineage="one",
        digest="digest",
        model="model",
        output=tmp_path / "calls",
    )
    async with httpx.AsyncClient(transport=adapter) as client:
        if ending == b"data: [DONE]\n\n":
            response = await client.post("http://native/v1/chat/completions", json={"model": "model", "stream": True})
            assert response.content == FIRST + b"data: [DONE]\n\n"
        else:
            with pytest.raises((asyncio.CancelledError, RuntimeError)):
                await client.post("http://native/v1/chat/completions", json={"model": "model", "stream": True})
            with pytest.raises(RuntimeError, match="unresolved"):
                await client.post("http://native/v1/chat/completions", json={"model": "model"})
    assert raw.released and not adapter.lock.locked()
    assert len(calls) == 1
    assert calls[0]["_max_num_tries"] == 1 and calls[0]["_resend_after_disconnect"] is False
    saved = json.loads((tmp_path / "calls/000000.chunk000000.json").read_text())
    assert bytes.fromhex(saved["hex"]) == FIRST
    assert (tmp_path / "calls/000000.complete.json").exists() == (ending == b"data: [DONE]\n\n")
