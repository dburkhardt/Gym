# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import tempfile
from pathlib import Path

import aiohttp
import pytest

from benchmarks.bioagentbench import execution


@pytest.mark.asyncio
async def test_actual_unix_relay_auth_route_stream_and_cleanup(monkeypatch):
    calls = []
    release = asyncio.Event()

    class Response:
        status = 200
        headers = {"Content-Type": "text/event-stream"}
        released = False

        @property
        def content(self):
            return self

        async def iter_chunked(self, size):
            yield b"data: first\n\n"
            await release.wait()
            yield b"data: [DONE]\n\n"

        def release(self):
            self.released = True

    upstream = Response()

    async def send(*args, **kwargs):
        calls.append((args, kwargs))
        return upstream

    monkeypatch.setattr(execution, "request", send)
    with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
        socket = Path(temporary) / "relay.sock"
        async with execution.interception_relay(
            endpoint="http://127.0.0.1:12345/v1", secret="local-only", socket_path=socket
        ):
            async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=str(socket))) as client:
                async with client.post("http://local/v1/chat/completions", json={}) as rejected:
                    assert rejected.status == 401
                async with client.post(
                    "http://local/arbitrary", json={}, headers={"Authorization": "Bearer local-only"}
                ) as rejected:
                    assert rejected.status == 404
                assert calls == []
                async with client.post(
                    "http://local/v1/chat/completions",
                    json={"model": "fixture"},
                    headers={"Authorization": "Bearer local-only"},
                ) as response:
                    first = await asyncio.wait_for(response.content.readuntil(b"\n\n"), 1)
                    assert first == b"data: first\n\n" and not release.is_set()
                    release.set()
                    assert await response.read() == b"data: [DONE]\n\n"
        assert not socket.exists() and upstream.released
        assert len(calls) == 1 and calls[0][0][1] == "http://127.0.0.1:12345/v1/chat/completions"
        assert calls[0][1]["_resend_after_disconnect"] is False


def test_real_relay_publishes_complete_port_file(tmp_path):
    import json
    import subprocess
    import sys
    import time

    from benchmarks.bioagentbench import relay

    port = tmp_path / "port.json"
    child = subprocess.Popen(
        [
            sys.executable,
            str(Path(relay.__file__)),
            "--socket",
            str(tmp_path / "unused.sock"),
            "--port-file",
            str(port),
        ]
    )
    try:
        deadline = time.monotonic() + 5
        while not port.exists():
            assert child.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        value = json.loads(port.read_text())
        assert type(value["port"]) is int and 1 <= value["port"] <= 65535
        assert not port.with_name("port.json.ready").exists()
    finally:
        child.terminate()
        child.wait(timeout=5)
