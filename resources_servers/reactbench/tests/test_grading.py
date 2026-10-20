# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.reactbench.app import Config, Server
from resources_servers.reactbench.grading import grade
from resources_servers.superchem.tests.test_native_contracts import response


@pytest.mark.parametrize("text,expected", [("2", 1), (" 2\n", 1), ("two", 0), ("", 0), ("The answer is 2", 0)])
def test_score(text, expected):
    assert grade(text, answer="2")["reward"] == expected


def test_native():
    cfg = Config(host="127.0.0.1", port=8080, entrypoint="app.py", name="reactbench", domain="knowledge")
    server = Server(config=cfg, server_client=MagicMock(spec=ServerClient))
    result = TestClient(server.setup_webserver()).post(
        "/verify",
        json={
            "responses_create_params": {"input": [{"role": "user", "content": "How many?"}]},
            "response": response("2"),
            "verifier_metadata": {"answer": "2"},
        },
    )
    assert result.status_code == 200, result.text
    assert result.json()["reward"] == 1
