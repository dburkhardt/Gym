# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from responses_api_agents.bixbench3.app import BixBench3Agent, BixBench3Request


def staged(tmp_path):
    root = tmp_path / "settings"
    root.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    settings = root / "case.json"
    settings.write_text(json.dumps({"task_id": "example", "model": "model-a", "run_root": str(run)}))
    (run / "grade.json").write_text(json.dumps({"score": 0.25, "artifact_count": 1, "artifacts": [{}]}))
    (run / "settings-sha256.txt").write_text(hashlib.sha256(settings.read_bytes()).hexdigest())
    agent = BixBench3Agent.model_construct(config=SimpleNamespace(settings_root=str(root)))
    body = BixBench3Request(task_id="example", settings_file="case.json", responses_create_params={"input": []})
    return agent, body, settings


@pytest.mark.asyncio
async def test_completed_run_returns_artifact_score_without_reexecution(tmp_path):
    agent, body, _ = staged(tmp_path)
    result = await agent.run(body)
    assert result.reward == 0.25
    assert result.response.model == "model-a"
    assert result.process_judging == "not_run"


@pytest.mark.asyncio
async def test_changed_model_cannot_reuse_score(tmp_path):
    agent, body, settings = staged(tmp_path)
    data = json.loads(settings.read_text())
    data["model"] = "model-b"
    settings.write_text(json.dumps(data))
    with pytest.raises(HTTPException) as error:
        await agent.run(body)
    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_task_mismatch_rejected(tmp_path):
    agent, body, _ = staged(tmp_path)
    body.task_id = "other"
    with pytest.raises(HTTPException) as error:
        await agent.run(body)
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_settings_escape_rejected(tmp_path):
    agent, body, settings = staged(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text(settings.read_text())
    body.settings_file = "../outside.json"
    with pytest.raises(HTTPException) as error:
        await agent.run(body)
    assert error.value.status_code == 400
