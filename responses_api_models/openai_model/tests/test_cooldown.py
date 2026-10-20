# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from email.utils import formatdate
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientResponseError, ClientSession, web

from nemo_gym.server_utils import ServerClient
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig


def server():
    config = SimpleModelServerConfig(
        host="127.0.0.1",
        port=8081,
        name="test",
        entrypoint="",
        openai_base_url="http://unused",
        openai_api_key="unused",
        openai_model="test",
        max_concurrent_requests=1,
        upstream_max_num_tries=1,
        upstream_rate_limit_cooldown_seconds=0.03,
        upstream_retry_policy={"max_attempts": 2, "backoff_initial_seconds": 0},
    )
    return SimpleModelServer(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ["0.03", None, "garbage", "nan", "-1", "date"])
async def test_wait_and_no_held_slot(header):
    model = server()
    calls = []

    async def operation():
        calls.append(time.monotonic())
        if len(calls) == 1:
            value = formatdate(time.time() + 1.5, usegmt=True) if header == "date" else header
            raise ClientResponseError(None, (), status=429, headers={"Retry-After": value} if value else {})
        return "ok"

    task = asyncio.create_task(model._call_upstream(operation))
    await asyncio.sleep(0.01)
    assert not model._semaphore.locked()
    assert await task == "ok"
    assert calls[1] - calls[0] >= 0.02


@pytest.mark.asyncio
async def test_loopback_shared_wait_and_ambiguous_failure():
    model = server()
    calls = []

    async def handler(request):
        calls.append(time.monotonic())
        if len(calls) == 1:
            return web.Response(status=429, headers={"Retry-After": "0.06"})
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_post("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        async with ClientSession() as client:

            async def operation():
                async with client.post(f"http://127.0.0.1:{port}/") as response:
                    response.raise_for_status()
                    return await response.text()

            first = asyncio.create_task(model._call_upstream(operation))
            while model._rate_limit_until == 0:
                await asyncio.sleep(0.001)
            second = asyncio.create_task(model._call_upstream(operation))
            assert await asyncio.gather(first, second) == ["ok", "ok"]
            assert len(calls) == 3
            assert min(calls[1:]) - calls[0] >= 0.05
        attempts = []

        async def uncertain():
            attempts.append(1)
            raise TimeoutError("accepted then response lost")

        with pytest.raises(TimeoutError):
            await model._call_upstream(uncertain)
        assert len(attempts) == 1
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_loopback_accepted_then_timeout_is_not_retried():
    from aiohttp import ClientTimeout

    model = server()
    accepted = []
    release = asyncio.Event()

    async def handler(request):
        accepted.append(1)
        await release.wait()
        return web.Response(text="late")

    app = web.Application()
    app.router.add_post("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        async with ClientSession(timeout=ClientTimeout(total=0.03)) as client:

            async def operation():
                async with client.post(f"http://127.0.0.1:{port}/") as response:
                    return await response.text()

            with pytest.raises(TimeoutError):
                await model._call_upstream(operation)
            assert accepted == [1]
            assert not model._semaphore.locked()
    finally:
        release.set()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 413, 500, 502])
async def test_non_429_never_retried_or_cooled_down(status):
    model = server()
    calls = []

    async def operation():
        calls.append(1)
        raise ClientResponseError(None, (), status=status, headers={"Retry-After": "100"})

    with pytest.raises(ClientResponseError):
        await model._call_upstream(operation)
    assert calls == [1]
    assert model._rate_limit_until == 0


@pytest.mark.asyncio
async def test_final_429_cools_other_requests_and_cancel_releases_wait():
    model = server()
    model.config.upstream_retry_policy.max_attempts = 1

    async def reject():
        raise ClientResponseError(None, (), status=429, headers={"Retry-After": "0.06"})

    with pytest.raises(ClientResponseError):
        await model._call_upstream(reject)
    called = []

    async def success():
        called.append(time.monotonic())
        return "ok"

    cancelled = asyncio.create_task(model._call_upstream(success))
    await asyncio.sleep(0.01)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert not called and not model._semaphore.locked()
    deadline = model._rate_limit_until
    assert await model._call_upstream(success) == "ok"
    assert called[0] >= deadline


@pytest.mark.asyncio
async def test_pool_timeout_does_not_dispatch_or_retry():
    model = server()
    model.config.upstream_pool_timeout_seconds = 0.01
    await model._semaphore.acquire()
    called = []

    async def operation():
        called.append(1)

    try:
        with pytest.raises(TimeoutError, match="acquiring"):
            await model._call_upstream(operation)
        assert called == []
        assert model._semaphore.locked()
    finally:
        model._semaphore.release()
