# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Orchestration tests; upstream metric correctness is tested separately."""

import json
from unittest.mock import AsyncMock

import pytest

from responses_api_agents.genotex import app


def response(output):
    return {
        "id": "response1",
        "created_at": 1.0,
        "model": "test-model",
        "object": "response",
        "output": output,
        "parallel_tool_calls": False,
        "tools": [],
        "tool_choice": "auto",
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in ["public", "prediction", "reference", "grading"]}
    monkeypatch.setattr(app, "stage", lambda *_: paths)
    container = AsyncMock()
    container.inspect.return_value = {"image_id": "fixture-image", "network_mode": "none"}
    container.shell.return_value = {"exit_code": 0, "output": "done"}
    monkeypatch.setattr(app, "Container", lambda **_: container)
    scorer = AsyncMock(return_value={"reward": 0.5, "raw_upstream": {"analysis": "fixture"}})
    monkeypatch.setattr(app, "score", scorer)
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())

    async def parsed(reply):
        return reply

    monkeypatch.setattr(app, "get_response_json", parsed)
    client = AsyncMock()
    config = app.GenoTEXConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="test",
        model_server={"type": "responses_api_models", "name": "policy_model"},
        data_root=str(tmp_path),
        work_root=str(tmp_path / "runs"),
    )
    agent = app.GenoTEXAgent.model_construct(config=config, server_client=client)
    body = app.GenoTEXRunRequest(
        responses_create_params={"input": [{"role": "user", "content": "task"}]},
        verifier_metadata={"trait": "Trait", "condition": None, "traits": ["Trait"], "task_id": "Trait__None"},
    )
    return agent, body, client, container, scorer


@pytest.mark.asyncio
async def test_tool_loop_saves_trajectory_and_scores(setup):
    agent, body, client, container, scorer = setup
    client.post.side_effect = [
        response(
            [
                {
                    "type": "function_call",
                    "name": "shell",
                    "arguments": json.dumps({"command": "echo done"}),
                    "call_id": "call1",
                    "id": "fc1",
                }
            ]
        ),
        response([]),
    ]
    result = await agent.run(body)
    assert result.reward == 0.5
    assert not result.mask_sample
    assert result.verifier_metadata["task_id"] == "Trait__None"
    container.shell.assert_awaited_once_with("echo done")
    container.stop.assert_awaited_once()
    scorer.assert_awaited_once()
    second = client.post.call_args_list[1].kwargs["json"]
    assert second["input"][-1]["type"] == "function_call_output"
    assert second["input"][-1]["call_id"] == "call1"


@pytest.mark.asyncio
async def test_model_failure_is_masked_infrastructure(setup):
    agent, body, client, container, scorer = setup
    client.post.side_effect = RuntimeError("endpoint unavailable")
    result = await agent.run(body)
    assert result.mask_sample
    assert result.reward == 0
    assert "endpoint unavailable" in result.failure_reason
    scorer.assert_not_awaited()
    container.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_step_budget_scores_partial_outputs(setup):
    agent, body, client, container, scorer = setup
    agent.config.max_steps = 1
    client.post.return_value = response(
        [{"type": "function_call", "name": "shell", "arguments": "invalid json", "call_id": "call1", "id": "fc1"}]
    )
    result = await agent.run(body)
    assert result.failure_reason == "max_steps_exhausted"
    assert not result.mask_sample
    container.shell.assert_not_awaited()
    scorer.assert_awaited_once()
