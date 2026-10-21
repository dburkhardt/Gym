# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from resources_servers.csmed_ft.grading import grade, macro_metrics


def test_controls():
    assert grade("Included", answer="included")["reward"] == 1
    assert grade("Excluded", answer="included")["reward"] == 0
    assert grade("", answer="excluded")["reward"] == 0
    assert grade("maybe", answer="excluded")["reward"] == 0
    assert macro_metrics(["included", "excluded"], ["included", "excluded"])["macro_f1"] == 1
    assert macro_metrics(["included", "excluded"], ["excluded", "included"])["macro_f1"] == 0


def test_seed_rejects_invalid_input_with_identity_and_preserves_valid():
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    from nemo_gym.server_utils import ServerClient
    from resources_servers.csmed_ft.app import Config, Server

    server = Server(
        config=Config(host="127.0.0.1", port=8080, name="csmed", entrypoint="app.py", domain="knowledge"),
        server_client=MagicMock(spec=ServerClient),
    )
    client = TestClient(server.setup_webserver())
    for flag in [False, None]:
        result = client.post(
            "/seed_session",
            json={"verifier_metadata": {"task_id": "frozen-invalid", "review_id": "review", "input_valid": flag}},
        )
        assert result.status_code == 422
        assert result.json()["detail"]["task_id"] == "frozen-invalid"
        assert result.json()["detail"]["dispatch_status"] == "not_sent"
    assert client.post("/seed_session", json={"verifier_metadata": {"input_valid": True}}).status_code == 200


async def test_simple_agent_rejects_before_model(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import pytest
    from aiohttp import ClientResponseError
    from fastapi.testclient import TestClient

    from nemo_gym.server_utils import ServerClient
    from resources_servers.csmed_ft.app import Config, Server
    from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig, SimpleAgentRunRequest
    from responses_api_agents.simple_agent.tests.test_app import _mock_response

    resource = Server(
        config=Config(host="127.0.0.1", port=8080, name="csmed", entrypoint="app.py", domain="knowledge"),
        server_client=MagicMock(spec=ServerClient),
    )
    client = TestClient(resource.setup_webserver())
    calls = []

    async def post(*, server_name, url_path, **kwargs):
        calls.append((server_name, url_path))
        assert url_path == "/seed_session"
        r = client.post(url_path, json=kwargs["json"])
        from types import SimpleNamespace

        reply = _mock_response(status=r.status_code, content=r.text)
        reply.raise_for_status.side_effect = ClientResponseError(
            request_info=SimpleNamespace(
                url="http://resource/seed_session", real_url="http://resource/seed_session", method="POST", headers={}
            ),
            history=(),
            status=r.status_code,
            message="invalid source",
        )
        return reply

    sc = MagicMock(spec=ServerClient)
    sc.post = AsyncMock(side_effect=post)
    agent = SimpleAgent(
        config=SimpleAgentConfig(
            host="127.0.0.1",
            port=8081,
            name="agent",
            entrypoint="app.py",
            resources_server={"type": "resources_servers", "name": "csmed"},
            model_server={"type": "responses_api_models", "name": "policy_model"},
            max_steps=1,
        ),
        server_client=sc,
    )
    body = SimpleAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "frozen input"}]},
            "verifier_metadata": {"input_valid": False, "task_id": "invalid-frozen"},
            "_ng_task_index": 3,
            "_ng_rollout_index": 0,
        }
    )
    with pytest.raises(ClientResponseError) as error:
        await agent.run(MagicMock(cookies={}, path_params={}), body)
    assert b"invalid-frozen" in error.value.response_content
    assert calls == [("csmed", "/seed_session")]


def test_modern_seed_missing_or_malformed_metadata():
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    from nemo_gym.server_utils import ServerClient
    from resources_servers.csmed_ft.app import Config, Server

    server = Server(
        config=Config(host="127.0.0.1", port=8080, name="csmed", entrypoint="app.py", domain="knowledge"),
        server_client=MagicMock(spec=ServerClient),
    )
    client = TestClient(server.setup_webserver())
    envelope = {
        "resources_session_id": "resource-1",
        "episode_id": {"rollout_id": "task-0"},
        "task_id": {"taskset": "frozen", "task_id": "missing"},
    }
    for data in [{}, {"verifier_metadata": None}, {"verifier_metadata": []}]:
        reply = client.post("/seed_session", json=envelope | {"task_data": data})
        assert reply.status_code == 422
        assert reply.json()["detail"]["failure_kind"] == "csmed_ft:invalid_input"
    reply = client.post("/seed_session", json=envelope | {"task_data": {"verifier_metadata": {"input_valid": True}}})
    assert reply.status_code == 200
    assert reply.json()["resources_session_id"] == "resource-1"


def test_creator_chunk_protocol_is_explicit_and_parser_differs():
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    from benchmarks.csmed_ft.creator_protocol import PROTOCOL
    from nemo_gym.server_utils import ServerClient
    from resources_servers.csmed_ft.app import Config, Server

    server = Server(
        config=Config(
            host="127.0.0.1",
            port=8080,
            name="csmed",
            entrypoint="app.py",
            domain="knowledge",
            scoring_protocol="creator-chunk-labels",
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    client = TestClient(server.setup_webserver())
    meta = {"input_valid": True, "answer": "included", "task_id": "paper:0"}
    assert client.post("/seed_session", json={"verifier_metadata": meta}).status_code == 422
    meta.update(protocol=PROTOCOL, recipe_sha256="fixture")
    assert client.post("/seed_session", json={"verifier_metadata": meta}).status_code == 200
    response = {
        "id": "fixture",
        "created_at": 0,
        "object": "response",
        "status": "completed",
        "model": "fixture",
        "output": [
            {
                "id": "message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "yes", "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }
    result = client.post(
        "/verify", json={"response": response, "verifier_metadata": meta, "responses_create_params": {"input": []}}
    )
    assert result.status_code == 200, result.text
    assert result.json()["reward"] == 1
    assert result.json()["diagnostics"]["metric_scope"] == "chunk_only_not_paper_accuracy"
    assert grade("yes", answer="included")["reward"] == 0
