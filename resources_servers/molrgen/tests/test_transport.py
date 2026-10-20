# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.molrgen import app
from resources_servers.superchem.tests.test_native_contracts import response


@pytest.mark.parametrize(
    "receipt,masked",
    [
        ({"reward": 0.5, "error": None, "meta": {"properties": ["QED"]}}, False),
        ({"reward": 0, "error": "docking failed", "meta": {}}, True),
        ({"reward": True}, True),
        ({"reward": None}, True),
    ],
)
@pytest.mark.parametrize("incomplete", [False, True])
def test_native_transport(monkeypatch, receipt, masked, incomplete):
    received = AsyncMock()
    received.__aenter__.return_value = received
    received.raise_for_status = MagicMock()
    received.json.return_value = receipt
    client = AsyncMock(return_value=received)
    monkeypatch.setattr(app, "request", client)
    cfg = app.Config(
        host="127.0.0.1",
        port=8080,
        name="molrgen",
        entrypoint="app.py",
        domain="knowledge",
        creator_url="http://creator",
    )
    server = app.Server(config=cfg, server_client=MagicMock(spec=ServerClient))
    metadata = {"properties": ["QED"], "objectives": ["maximize"], "target": [0]}
    completion = response(r"\boxed{CCO}")
    if incomplete:
        completion["status"] = "incomplete"
        completion["incomplete_details"] = {"reason": "max_output_tokens"}
    result = TestClient(server.setup_webserver()).post(
        "/verify",
        json={
            "responses_create_params": {"input": [{"role": "user", "content": "Generate"}]},
            "response": completion,
            "verifier_metadata": {"metadata": metadata},
        },
    )
    assert result.status_code == 200, result.text
    assert result.json()["mask_sample"] == masked
    if masked:
        assert result.json()["failure_kind"] == "verifier_error"
    else:
        assert result.json()["reward"] == receipt["reward"]
        assert result.json()["failure_kind"] == ("model_budget" if incomplete else None)
    assert result.json()["diagnostics"]["creator_response"] == receipt
    assert client.call_args.kwargs["json"]["metadata"] == [metadata]

    client.assert_awaited_once()
    assert client.call_args.kwargs["_max_num_tries"] == 1
    assert client.call_args.kwargs["_resend_after_disconnect"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_transport_failure_does_not_repeat_scientific_work(monkeypatch, cancelled):
    import asyncio
    from types import SimpleNamespace

    error = asyncio.CancelledError() if cancelled else asyncio.TimeoutError("creator outcome unknown")
    transport = AsyncMock(side_effect=error)
    monkeypatch.setattr(app, "request", transport)
    verifier = app.Verifier()
    verifier.config = SimpleNamespace(creator_url="http://creator", creator_timeout_seconds=1)
    body = app.Request.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Generate"}]},
            "response": response(r"\boxed{CCO}"),
            "verifier_metadata": {"metadata": {"properties": ["QED"]}},
        }
    )
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await verifier.verify(body)
    else:
        result = await verifier.verify(body)
        assert result.mask_sample and result.failure_kind == "verifier_error"
        assert result.diagnostics["reward_is_serialization_placeholder"]
    transport.assert_awaited_once()
    assert transport.call_args.kwargs["_max_num_tries"] == 1
    assert transport.call_args.kwargs["_resend_after_disconnect"] is False
