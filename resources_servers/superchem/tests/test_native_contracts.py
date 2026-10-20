# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate real Config and POST /verify contracts for campaign resources."""

import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient


def response(text: str, model: str = "nvidia/nvidia/nemotron-3.5-super-vl") -> dict:
    return {
        "id": "fixture",
        "created_at": 0,
        "model": model,
        "object": "response",
        "output": [
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
    }


@pytest.mark.parametrize(
    "name,text,metadata",
    [
        ("superchem", r"\boxed{A}", {"answer": ["A"], "question_type": "multiple_choice"}),
        ("biomaze", "Yes", {"answer": "Yes"}),
        ("microvqa", "The answer is (2)", {"correct_index": 1}),
        ("moleculeqa", "B", {"answer": "B"}),
        (
            "evidencebench",
            "DECISION: [0]",
            {"sentence_index2aspects": {"0": ["a"]}, "limit": 10, "candidate_count": 1},
        ),
        ("csmed_ft", "Included", {"answer": "included", "input_valid": True}),
    ],
)
def test_native_verify(name, text, metadata):
    mod = importlib.import_module(f"resources_servers.{name}.app")
    config = mod.Config(host="127.0.0.1", port=8080, entrypoint="app.py", name=name, domain="knowledge")
    server = mod.Server(config=config, server_client=MagicMock(spec=ServerClient))
    app = server.setup_webserver()
    with TestClient(app) as client:
        result = client.post(
            "/verify",
            json={
                "responses_create_params": {"input": [{"role": "user", "content": "fixture"}]},
                "response": response(text),
                "verifier_metadata": metadata,
            },
        )
    assert result.status_code == 200, result.text
    assert result.json()["reward"] == 1.0


@pytest.mark.parametrize(
    "name,metadata,judge_answer",
    [
        ("biomaze", {"protocol": "openended", "question": "q", "answer": "r"}, "Judge: Correct"),
        ("fdarxbench", {"question": "q", "answer": "r"}, "A"),
    ],
)
def test_native_judge_contract(name, metadata, judge_answer, tmp_path, monkeypatch):
    from nemo_gym.openai_utils import NeMoGymResponse

    mod = importlib.import_module(f"resources_servers.{name}.app")
    template = tmp_path / "prompt.txt"
    template.write_text("{question} {target} {predicted_answer}" if name == "fdarxbench" else "Creator rubric")
    config = mod.Config(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name=name,
        domain="knowledge",
        judge_model_server={"type": "responses_api_models", "name": "policy_model"},
        judge_prompt_file=str(template),
    )
    judge = AsyncMock(return_value=NeMoGymResponse.model_validate(response(judge_answer)))
    monkeypatch.setattr(mod, "call_judge", judge)
    server = mod.Server(config=config, server_client=MagicMock(spec=ServerClient))
    with TestClient(server.setup_webserver()) as client:
        result = client.post(
            "/verify",
            json={
                "responses_create_params": {"input": [{"role": "user", "content": "fixture"}]},
                "response": response("candidate"),
                "verifier_metadata": metadata,
            },
        )
    assert result.status_code == 200, result.text
    assert result.json()["reward"] == 1.0
    assert judge.call_args.kwargs["json"].model == "nvidia/nvidia/nemotron-3.5-super-vl"
    assert result.json()["diagnostics"]["judge_response"]["model"] == "nvidia/nvidia/nemotron-3.5-super-vl"
