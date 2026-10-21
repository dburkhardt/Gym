# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real verifier and shared judge binding with an offline HTTP fixture."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from omegaconf import OmegaConf

from nemo_gym.judge import JudgeError
from resources_servers.fdarxbench.app import Request, Verifier


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
async def test_actual_guarded_verifier_identity_and_protocol(tmp_path):
    seen = []

    async def endpoint(scope, receive, send):
        body = await receive()
        seen.append((json.loads(body["body"]), dict(scope["headers"])))
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send(
            {"type": "http.response.body", "body": json.dumps(response("A: Correct reference match.")).encode()}
        )

    prompt = tmp_path / "rubric.txt"
    template = (
        Path(os.environ["FDARX_TEST_RUBRIC"]).read_text()
        if os.environ.get("FDARX_TEST_RUBRIC")
        else "Question: {question}\nReference: {target}\nCandidate: {predicted_answer}"
    )
    prompt.write_text(template)
    verifier = Verifier()
    verifier.config = SimpleNamespace(judge_model_server=SimpleNamespace(name="judge"), judge_prompt_file=str(prompt))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=endpoint), base_url="http://fixture") as http:
        verifier.server_client = Client(http)
        base = {
            "responses_create_params": {"input": [{"role": "user", "content": "question"}]},
            "response": response("candidate"),
            "verifier_metadata": {"question": "question", "answer": "reference"},
            "_ng_task_index": 7,
            "_ng_rollout_index": 2,
        }
        with pytest.raises(JudgeError, match="persisted verification_id"):
            await verifier.verify(Request.model_validate(base))
        assert seen == []
        for identity in ["attempt-a", "attempt-a", "attempt-b"]:
            result = await verifier.verify(Request.model_validate(base | {"verification_id": identity}))
            assert result.reward == 1 and result.model_dump()["_ng_task_index"] == 7
        saved = result.model_dump(mode="json")
        preserved = json.loads(json.dumps(saved))
        replay = await verifier.verify(Request.model_validate(saved))
        assert replay.reward == result.reward and replay.diagnostics == result.diagnostics
        assert replay.model_dump()["_ng_task_index"] == 7 and saved == preserved
        failed = verifier.judge_failure(Request.model_validate(saved), result.response, "fixture judge failure")
        failed_saved = failed.model_dump(mode="json")
        repeated_failure = verifier.judge_failure(
            Request.model_validate(failed_saved), result.response, "new judge failure"
        )
        assert repeated_failure.mask_sample and repeated_failure.failure_reason == "new judge failure"
        assert failed_saved["failure_reason"] == "fixture judge failure"
        assert repeated_failure.model_dump()["_ng_task_index"] == 7
        assert seen[0] == seen[1]
        assert seen[0][1][b"x-nemo-gym-logical-call-id"] != seen[2][1][b"x-nemo-gym-logical-call-id"]
        payload = seen[0][0]
        assert payload["temperature"] == 0 and payload["max_output_tokens"] == 4096 and payload["model"] == MODEL
        assert payload["input"][1]["content"] == template.format(
            question="question", target="reference", predicted_answer="candidate"
        )
        previous_digest = seen[-1][1][b"x-nemo-gym-call-config-sha256"]
        prompt.write_text(prompt.read_text() + "\nchanged rubric")
        await verifier.verify(Request.model_validate(base | {"verification_id": "attempt-b"}))
        assert seen[-1][1][b"x-nemo-gym-call-config-sha256"] != previous_digest


def test_config_keeps_judge_separate_from_solver(monkeypatch):
    monkeypatch.setenv("FDARX_JUDGE_PROMPT", "/explicit/pinned-rubric.txt")
    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.load(root / "resources_servers/fdarxbench/configs/fdarxbench.yaml")
    resource = config.fdarxbench.resources_servers.fdarxbench
    assert resource.judge_model_server.name == "judge_model"
    config.fdarxbench_simple_agent.responses_api_agents.simple_agent.model_server.name = "comparison_solver"
    assert resource.judge_model_server.name == "judge_model"
    assert resource.judge_prompt_file == "/explicit/pinned-rubric.txt"
    benchmark = OmegaConf.load(root / "benchmarks/fdarxbench/config.yaml")
    role = benchmark.fdarxbench.resources_servers.fdarxbench.datasets[0].required_model_roles.judge
    assert role.component == "fdarxbench" and role.binding_field == "judge_model_server"
    assert list(role.capabilities) == ["text_generation"]
