# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real loopback provider, actual model handler/guard/capture and caller cancellation."""

import asyncio
import json
import socket
import time
from unittest.mock import MagicMock

import httpx
import pytest
import uvicorn
from aiohttp import ClientSession, web

from nemo_gym import server_utils
from nemo_gym.base_responses_api_model import CaptureStore
from nemo_gym.server_utils import ServerClient
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "rejected_then_success",
        "deadline_wait",
        "accepted_timeout",
        "admission_before_send",
        "admission_after_rejection",
    ],
)
async def test_guarded_cooldown_physical_capture_and_deadline(tmp_path, monkeypatch, mode):
    calls = []

    async def provider(request):
        calls.append((time.monotonic(), await request.json()))
        if mode == "accepted_timeout":
            await asyncio.sleep(0.2)
        elif len(calls) == 1:
            return web.json_response(
                {"error": {"message": "capacity"}},
                status=429,
                headers={"Retry-After": "0.2" if mode in {"deadline_wait", "admission_after_rejection"} else "0.02"},
            )
        return web.json_response(
            {
                "id": "response-fixture",
                "created_at": 1,
                "model": "fixture-model",
                "object": "response",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "status": "completed",
            }
        )

    upstream = web.Application()
    upstream.router.add_post("/v1/responses", provider)
    runner = web.AppRunner(upstream)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    config = SimpleModelServerConfig(
        host="127.0.0.1",
        port=1,
        name="policy_model",
        entrypoint="app.py",
        openai_base_url=f"http://127.0.0.1:{port}/v1",
        openai_api_key="fixture",
        openai_model="fixture-model",
        max_concurrent_requests=1,
        upstream_max_num_tries=1,
        upstream_request_timeout_seconds=0.03 if mode == "accepted_timeout" else 1,
        upstream_rate_limit_cooldown_seconds=0.02,
        upstream_retry_policy={"max_attempts": 3, "backoff_initial_seconds": 0},
    )
    global_config = {
        "observability_enabled": True,
        "provider_dispatch_guard": True,
        "model_call_capture_dir": str(tmp_path / "capture"),
    }
    model = SimpleModelServer(
        config=config, server_client=MagicMock(spec=ServerClient, global_config_dict=global_config)
    )
    if mode == "accepted_timeout":
        model.config.upstream_admission_timeout_seconds = 0.01
    if mode.startswith("admission_"):
        model.config.upstream_admission_timeout_seconds = 0.03
    if mode == "admission_before_send":
        model._rate_limit_until = time.monotonic() + 0.2
    app = model.setup_webserver()
    model.setup_exception_middleware(app)
    model.setup_cancellation_middleware(app)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    model_port = listener.getsockname()[1]
    host = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    serving = asyncio.create_task(host.serve(sockets=[listener]))
    while not host.started:
        await asyncio.sleep(0.001)
    try:
        async with ClientSession() as physical:
            monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: physical)
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{model_port}") as caller:

                async def invoke():
                    return await caller.post(
                        "/ng-rollout/0-0/v1/responses",
                        json={"input": "unchanged prompt"},
                        headers={
                            "x-nemo-gym-logical-call-id": "fixed-call",
                            "x-nemo-gym-call-config-sha256": "a" * 64,
                        },
                    )

                if mode == "deadline_wait":
                    with pytest.raises(TimeoutError):
                        async with asyncio.timeout(0.08):
                            await invoke()
                    await asyncio.sleep(0.25)
                else:
                    reply = await invoke()
                    assert reply.status_code == (200 if mode == "rejected_then_success" else 500)
            assert len(calls) == (
                0 if mode == "admission_before_send" else 2 if mode == "rejected_then_success" else 1
            )
            if len(calls) == 2:
                assert calls[0][1] == calls[1][1]
                assert calls[1][0] - calls[0][0] >= 0.02
            for _ in range(100):
                if list((tmp_path / "capture").glob("*.capture.jsonl")):
                    break
                await asyncio.sleep(0.01)
            records = [
                json.loads(line)
                for p in (tmp_path / "capture").glob("*.capture.jsonl")
                for line in p.read_text().splitlines()
            ]
            if mode == "rejected_then_success":
                assert len(records) == 1
                attempts = records[0]["provider_attempts"]
                assert [a["outcome"] for a in attempts] == ["rejected", "captured"]
                assert len({a["attempt_id"] for a in attempts}) == 2
            elif mode == "deadline_wait":
                assert len(records) == 1
                assert records[0]["error_category"] == "cancelled"
                assert len(records[0]["provider_attempts"]) == 1
                assert records[0]["provider_attempts"][0]["outcome"] == "rejected"
                assert records[0]["provider_attempts"][0]["status_code"] == 429
                assert not list((tmp_path / "capture").glob("*.pending"))
            elif mode.startswith("admission_"):
                assert len(records) == 1
                attempts = records[0]["provider_attempts"]
                assert len(attempts) == (0 if mode == "admission_before_send" else 1)
                if attempts:
                    assert attempts[0]["outcome"] == "rejected"
                    assert attempts[0]["status_code"] == 429
                assert records[0]["error_category"] is not None
                assert "UpstreamAdmissionTimeoutError" in reply.text
                assert not CaptureStore(tmp_path / "capture", provider_dispatch_guard=True).has_pending("0-0")
            elif mode == "accepted_timeout":
                store = CaptureStore(tmp_path / "capture", provider_dispatch_guard=True)
                assert store.has_pending("0-0")
                assert len(records) == 1
                assert records[0]["error_category"] == "timeout"
                assert len(records[0]["provider_attempts"]) == 1
                assert records[0]["provider_attempts"][0]["outcome"] == "unknown"
                [pending] = (tmp_path / "capture").glob("*.pending")
                assert json.loads(pending.read_text())["provider_attempts"] == records[0]["provider_attempts"]
            assert not model._semaphore.locked()
    finally:
        host.should_exit = True
        await serving
        listener.close()
        await runner.cleanup()
