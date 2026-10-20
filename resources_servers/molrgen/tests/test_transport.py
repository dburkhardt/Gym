# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock

import httpx
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
    ],
)
def test_native_transport(monkeypatch, receipt, masked):
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post.return_value = httpx.Response(
        200, json=receipt, request=httpx.Request("POST", "http://creator/get_reward")
    )
    monkeypatch.setattr(app.httpx, "AsyncClient", lambda **kwargs: client)
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
    result = TestClient(server.setup_webserver()).post(
        "/verify",
        json={
            "responses_create_params": {"input": [{"role": "user", "content": "Generate"}]},
            "response": response(r"\boxed{CCO}"),
            "verifier_metadata": {"metadata": metadata},
        },
    )
    assert result.status_code == 200, result.text
    assert result.json()["mask_sample"] == masked
    assert result.json()["diagnostics"]["creator_response"] == receipt
    assert client.post.call_args.kwargs["json"]["metadata"] == [metadata]
