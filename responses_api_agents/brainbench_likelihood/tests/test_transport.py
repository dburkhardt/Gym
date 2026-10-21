# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request, Response
from fastapi.testclient import TestClient
from pydantic import ValidationError

from benchmarks.brainbench.protocol import MODEL
from nemo_gym.base_responses_api_model import CaptureStore, read_model_call_records
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.provider_dispatch import current_provider_dispatch
from nemo_gym.server_utils import ServerClient
from responses_api_agents.brainbench_likelihood.app import BrainBenchAgent, Config
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "partial_usage", "missing_logprobs"])
async def test_actual_model_handler_capture_and_two_call_identity(tmp_path, failure):
    global_config = {
        "provider_dispatch_guard": True,
        "observability_enabled": True,
        "model_call_capture_dir": str(tmp_path / "capture"),
    }
    model = SimpleModelServer(
        config=SimpleModelServerConfig(
            host="127.0.0.1",
            port=1,
            name="policy",
            entrypoint="app.py",
            openai_base_url="http://unused.invalid",
            openai_api_key="fixture",
            openai_model=MODEL,
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict=global_config),
    )
    requests = []

    async def provider(**payload):
        dispatch = current_provider_dispatch.get()
        dispatch.before_send()
        dispatch.response_headers(200)
        requests.append(payload)
        usage = (
            {"prompt_tokens": 2}
            if failure == "partial_usage"
            else {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
        )
        result = {
            "id": "chat-" + str(len(requests)),
            "created": 1,
            "object": "chat.completion",
            "model": MODEL,
            "choices": [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": "x"}}],
            "usage": usage,
            "prompt_logprobs": [None, {"7": {"logprob": -float(len(requests))}}],
        }

        if failure == "missing_logprobs":
            result.pop("prompt_logprobs")
        return result

    model._client = SimpleNamespace(create_chat_completion=AsyncMock(side_effect=provider))
    client = TestClient(model.setup_webserver())

    async def post(**kwargs):
        assert kwargs["server_name"] == "policy"
        assert kwargs["timeout"].total == 180
        reply = client.post(kwargs["url_path"], json=kwargs["json"], headers=kwargs["headers"])
        assert reply.status_code == 200, reply.text
        return SimpleNamespace(ok=True, read=AsyncMock(return_value=reply.content), cookies={})

    agent = BrainBenchAgent(
        config=Config(
            host="127.0.0.1",
            port=2,
            name="brainbench",
            entrypoint="app.py",
            output_root=tmp_path / "attempts",
            resources_server={"type": "resources_servers", "name": "scorer"},
            model_server={"type": "responses_api_models", "name": "policy"},
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict=global_config),
    )
    agent.server_client.post = AsyncMock(side_effect=post)
    request = Request(
        {
            "type": "http",
            "path": "/ng-rollout/brainbench-test/v1/responses",
            "query_string": b"",
            "headers": [],
            "path_params": {"rollout_id": "brainbench-test"},
        }
    )
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "user", "content": json.dumps({"choices": ["original A", "original B"]})}]
    )
    if failure:
        with pytest.raises((ValidationError, KeyError)):
            await agent.responses(request, Response(), body)
        with pytest.raises(FileExistsError):
            await agent.responses(request, Response(), body)
        assert len(requests) == 1
        assert not list((tmp_path / "attempts").glob("*/result.json"))
        return
    result = await agent.responses(request, Response(), body)
    assert [x["messages"][0]["content"] for x in requests] == ["original A", "original B"]
    assert all(x["max_tokens"] == 1 and x["prompt_logprobs"] == 0 for x in requests)
    assert json.loads(result.output_text)["mean_nll"] == [1, 2]
    assert [x["response_id"] for x in result.ng_trajectory["invocations"][0]["model_calls"]] == ["chat-1", "chat-2"]
    assert len(result.ng_trajectory["turns"]) == 2
    if failure:
        assert result.usage is None
    else:
        assert result.usage.input_tokens == 4
        assert result.usage.input_tokens_details.cached_tokens is None
    first = next((tmp_path / "attempts").glob("*/0.intent.json"))
    saved = json.loads(first.read_text())
    replay = client.post(
        "/ng-rollout/brainbench-test/v1/chat/completions", json=saved["request"], headers=saved["headers"]
    )
    assert replay.status_code == 200 and replay.json()["id"] == "chat-1"
    assert len(requests) == 2
    captured = read_model_call_records(
        CaptureStore(tmp_path / "capture", provider_dispatch_guard=True), "brainbench-test"
    )
    assert len(captured) == 3
    assert sum(x.replayed_from_model_call_id is not None for x in captured) == 1
    with pytest.raises(FileExistsError):
        await agent.responses(request, Response(), body)
    assert len(requests) == 2
    assert len(list((tmp_path / "attempts").glob("*/attempt.json"))) == 1
