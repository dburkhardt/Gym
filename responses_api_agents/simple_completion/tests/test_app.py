# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request, Response

from benchmarks.scienceqa.creator_recipe import PROTOCOL
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.scienceqa_creator_recipe.app import Config as ResourceConfig
from resources_servers.scienceqa_creator_recipe.app import Server, VerifyRequest
from responses_api_agents.simple_completion.app import Config, SimpleCompletionAgent


def agent(tmp_path, text="The answer is B.", finish="stop"):
    payload = {
        "id": "completion1",
        "object": "text_completion",
        "created": 1,
        "model": "fixture",
        "choices": [{"index": 0, "text": text, "finish_reason": finish}],
    }
    client = MagicMock(spec=ServerClient, global_config_dict={"provider_dispatch_guard": True})
    client.post = AsyncMock(
        return_value=SimpleNamespace(
            status=200,
            ok=True,
            read=AsyncMock(return_value=json.dumps(payload).encode()),
            cookies={},
            raise_for_status=lambda: None,
        )
    )
    config = Config(
        host="127.0.0.1",
        port=1,
        name="agent",
        entrypoint="app.py",
        output_root=tmp_path,
        model_name="fixture",
        model_server={"type": "responses_api_models", "name": "policy"},
        resources_server={"type": "resources_servers", "name": "resource"},
    )
    return SimpleCompletionAgent(config=config, server_client=client)


@pytest.mark.asyncio
async def test_literal_and_reference_boundary_and_reverify(tmp_path):
    server = agent(tmp_path)
    literal = "Question: demo\nAnswer:"
    request = Request(
        {
            "type": "http",
            "path": "/ng-rollout/fixture/v1/responses",
            "query_string": b"",
            "headers": [],
            "path_params": {"rollout_id": "fixture"},
        }
    )
    params = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {
                "role": "user",
                "content": json.dumps({"prompt": literal, "max_tokens": 512, "temperature": 0.0, "stop": ["\n"]}),
            }
        ]
    )
    result = await server.responses(request, Response(), params)
    call = server.server_client.post.call_args.kwargs
    assert call["url_path"].endswith("/v1/completions")
    assert call["json"]["prompt"] == literal
    assert "answer" not in call["json"] and "verifier_metadata" not in call["json"]
    assert result.usage is None and result.output_text == "The answer is B."
    assert len(result.model_dump()["ng_trajectory"]["turns"]) == 1
    with pytest.raises(FileExistsError):
        await server.responses(request, Response(), params)
    assert server.server_client.post.await_count == 1
    resource = Server(
        config=ResourceConfig(host="127.0.0.1", port=2, name="resource", entrypoint="app.py"),
        server_client=server.server_client,
    )
    body = VerifyRequest(
        task_id="1",
        response=result,
        responses_create_params=params,
        verifier_metadata={"protocol": PROTOCOL, "choices": ["x", "y"], "answer": 1},
    )
    scored = await resource.verify(body)
    assert scored.reward == 1 and scored.scoring_details["creator_raw_accuracy"] == 1
    repeated = await resource.verify(VerifyRequest.model_validate(scored.model_dump()))
    assert repeated.scoring_details == scored.scoring_details


@pytest.mark.asyncio
async def test_fallback_uncomputed_and_length_preserved(tmp_path):
    server = agent(tmp_path, "", "length")
    params = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "user", "content": json.dumps({"prompt": "literal", "max_tokens": 512})}]
    )
    result = await server.responses(
        Request(
            {
                "type": "http",
                "path": "/ng-rollout/fixture/v1/responses",
                "query_string": b"",
                "headers": [],
                "path_params": {"rollout_id": "fixture"},
            }
        ),
        Response(),
        params,
    )
    assert result.status == "incomplete"
    assert result.incomplete_details.reason == "max_output_tokens"
    resource = Server(
        config=ResourceConfig(host="127.0.0.1", port=2, name="resource", entrypoint="app.py"),
        server_client=server.server_client,
    )
    scored = await resource.verify(
        VerifyRequest(
            task_id="1",
            response=result,
            responses_create_params=params,
            verifier_metadata={"protocol": PROTOCOL, "choices": ["x", "y"], "answer": 1},
        )
    )
    assert scored.mask_sample and scored.scoring_details["creator_raw_accuracy"] is None
    assert scored.scoring_details["strict_accuracy"] == 0
