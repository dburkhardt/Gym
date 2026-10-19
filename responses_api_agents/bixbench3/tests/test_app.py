# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from responses_api_agents.bixbench3.app import BixBench3Agent, BixBench3Request, terminate_process_tree
from responses_api_agents.bixbench3.runtime import run_fingerprint


def staged(tmp_path):
    root = tmp_path / "settings"
    root.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    (run / "dataset").mkdir()
    (run / "dataset" / "prompt.md").write_text("original prompt")
    data = tmp_path / "data"
    data.mkdir()
    gold = tmp_path / "gold"
    gold.mkdir()
    (gold / "spec.json").write_text("{}")
    settings = root / "case.json"
    settings.write_text(
        json.dumps(
            {
                "task_id": "example",
                "model": "model-a",
                "run_root": str(run),
                "data_dir": str(data),
                "grading_specification": str(gold / "spec.json"),
            }
        )
    )
    (run / "grade.json").write_text(json.dumps({"score": 0.25, "artifact_count": 1, "artifacts": [{}]}))
    (run / "settings-sha256.txt").write_text(run_fingerprint(settings))
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


@pytest.mark.asyncio
async def test_changed_prompt_cannot_reuse_score(tmp_path):
    agent, body, settings = staged(tmp_path)
    run = Path(json.loads(settings.read_text())["run_root"])
    (run / "dataset" / "prompt.md").write_text("changed prompt")
    with pytest.raises(HTTPException) as error:
        await agent.run(body)
    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_termination_stops_grandchild_heartbeat(tmp_path):
    heartbeat = tmp_path / "heartbeat"
    child = (
        "import signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"p=Path({str(heartbeat)!r}); "
        "exec('while True:\\n p.write_text(str(time.time()))\\n time.sleep(0.02)')"
    )
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',sys.argv[1]]); time.sleep(60)"
    process = await asyncio.create_subprocess_exec(sys.executable, "-c", parent, child, start_new_session=True)
    try:
        for _ in range(100):
            if heartbeat.exists():
                break
            await asyncio.sleep(0.02)
        assert heartbeat.exists()
        await terminate_process_tree(process)
        await asyncio.sleep(0.1)
        before = heartbeat.read_text()
        await asyncio.sleep(0.1)
        assert heartbeat.read_text() == before
    finally:
        if process.returncode is None:
            await terminate_process_tree(process)
