# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real verifier and shared judge binding with an offline HTTP fixture."""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from omegaconf import OmegaConf

from nemo_gym.judge import JudgeError
from resources_servers.sciknoweval.app import (
    SciKnowEvalResourcesServer,
    SciKnowEvalResourcesServerConfig,
    SciKnowEvalVerifyRequest,
)


MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


def response(text):
    return {
        "id": "fixture",
        "created_at": 0,
        "object": "response",
        "model": MODEL,
        "status": "completed",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "id": "message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    }


class Client:
    global_config_dict = OmegaConf.create({"provider_dispatch_guard": True})

    def __init__(self, http):
        self.http = http

    async def post(self, *, server_name, url_path, json, headers):
        assert server_name == "judge"
        result = await self.http.post(url_path, json=json.model_dump(mode="json"), headers=headers)

        async def read():
            return result.content

        return SimpleNamespace(
            status=result.status_code,
            ok=result.is_success,
            read=read,
            content=SimpleNamespace(read=read),
            raise_for_status=result.raise_for_status,
        )


@pytest.mark.asyncio
async def test_guarded_task_rubric_http_identity_and_saved_reverify():
    seen = []

    async def endpoint(scope, receive, send):
        body = await receive()
        seen.append((json.loads(body["body"]), dict(scope["headers"])))
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": json.dumps(response("Rating: 4")).encode()})

    from unittest.mock import MagicMock

    from nemo_gym.server_utils import ServerClient

    server = SciKnowEvalResourcesServer(
        config=SciKnowEvalResourcesServerConfig(
            name="sciknoweval",
            host="127.0.0.1",
            port=8123,
            entrypoint="app.py",
            judge_model_server={"type": "responses_api_models", "name": "judge"},
            judge_responses_create_params={"input": [], "temperature": 0, "max_output_tokens": 16384},
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=endpoint), base_url="http://fixture") as http:
        server.server_client.post = Client(http).post
        server.server_client.global_config_dict = Client.global_config_dict
        base = {
            "responses_create_params": {"input": [{"role": "user", "content": "Question"}]},
            "response": response("candidate"),
            "_ng_task_index": 7,
            "_ng_rollout_index": 2,
            "verifier_metadata": {
                "answer_type": "open-ended-qa",
                "expected_answer": "gold",
                "judge_system": "rubric",
                "judge_prefix": "Gold: gold. Candidate: ",
                "judge_suffix": " Grade.",
                "judge_scale": "score",
            },
        }
        with pytest.raises(JudgeError, match="persisted verification_id"):
            await server.verify(SciKnowEvalVerifyRequest.model_validate(base))
        assert not seen
        for identity in ["saved-a", "saved-a", "fresh-b"]:
            result = await server.verify(SciKnowEvalVerifyRequest.model_validate(base | {"verification_id": identity}))
            assert result.reward == 0.75 and result.model_dump()["_ng_task_index"] == 7
        assert seen[0] == seen[1]
        assert seen[0][1][b"x-nemo-gym-logical-call-id"] != seen[2][1][b"x-nemo-gym-logical-call-id"]
        assert [{"role": m["role"], "content": m["content"]} for m in seen[0][0]["input"]] == [
            {"role": "system", "content": "rubric"},
            {"role": "user", "content": "Gold: gold. Candidate: candidate Grade."},
        ]
        assert seen[0][0]["temperature"] == 0 and seen[0][0]["max_output_tokens"] == 16384
        saved = result.model_dump()
        assert (await server.verify(SciKnowEvalVerifyRequest.model_validate(saved))).reward == 0.75
        old = seen[-1][1][b"x-nemo-gym-call-config-sha256"]
        saved["verifier_metadata"]["judge_suffix"] += " Changed rubric."
        await server.verify(SciKnowEvalVerifyRequest.model_validate(saved))
        assert seen[-1][1][b"x-nemo-gym-call-config-sha256"] != old


def test_independent_judge_binding():
    config = OmegaConf.load(Path(__file__).resolve().parents[3] / "benchmarks/sciknoweval/config.yaml")
    resource = config.sciknoweval.resources_servers.sciknoweval
    assert resource.judge_model_server.name == "judge_model"
    role = resource.datasets[0].required_model_roles.judge
    assert role.component == "sciknoweval" and role.binding_field == "judge_model_server"
    assert list(role.capabilities) == ["text_generation"]
