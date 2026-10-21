# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual creator initialization and native seed-before-model contract (no scoring)."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.campaign_creator_scoring import app


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def creator():
    path = Path(os.environ.get("CHEMIQ_SOURCE", ROOT.parent / "chemiq-upstream"))
    if not (path / "questions/chemiq.jsonl").is_file():
        pytest.skip("Pinned ChemIQ checkout unavailable; set CHEMIQ_SOURCE")
    row = json.loads((path / "questions/chemiq.jsonl").read_text().splitlines()[0])
    return path, row["uuid"]


def worker(python, creator, tmp_path):
    repo, uuid = creator
    request = tmp_path / "request.json"
    output = tmp_path / "result.json"
    request.write_text(
        json.dumps({"benchmark": "chemiq", "repo": str(repo), "task": {"uuid": uuid}, "preflight": True})
    )
    result = subprocess.run(
        [
            str(python),
            str(ROOT / "resources_servers/campaign_creator_scoring/score_worker.py"),
            str(request),
            str(output),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result, output


def test_actual_worker_missing_rdkit(creator, tmp_path):
    # Use an actual isolated interpreter, not an import mock. CI may supply one.
    python = Path(os.environ.get("CHEMIQ_MISSING_RDKIT_PYTHON", sys.executable))
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.util; assert importlib.util.find_spec('rdkit') is None; import numpy, requests",
        ],
        capture_output=True,
    )
    if probe.returncode:
        pytest.skip("Provide CHEMIQ_MISSING_RDKIT_PYTHON with numpy/requests but no RDKit")
    result, output = worker(python, creator, tmp_path)
    assert result.returncode != 0
    assert "No module named 'rdkit'" in result.stderr
    assert not output.exists()


def test_actual_qualified_worker_initializes_only(creator, tmp_path):
    python = Path(
        os.environ.get("CHEMIQ_QUALIFIED_PYTHON", ROOT / "resources_servers/campaign_creator_scoring/.venv/bin/python")
    )
    if not python.exists():
        pytest.skip("Provide CHEMIQ_QUALIFIED_PYTHON")
    result, output = worker(python, creator, tmp_path)
    assert result.returncode == 0, result.stderr
    details = json.loads(output.read_text())["details"]
    assert details == {
        "status": "ready",
        "scope": "chemiq_initialization_only",
        "interpreter": str(python),
        "creator_path": str(creator[0] / "utils/answer_verifier.py"),
        "uuid": creator[1],
    }
    assert "reward" not in details


def resource(creator, benchmark="chemiq"):
    return app.CreatorResourcesServer(
        config=app.Config(
            host="127.0.0.1",
            port=8080,
            name="resource",
            entrypoint="app.py",
            benchmark=benchmark,
            upstream_repo=str(creator[0]),
        ),
        server_client=MagicMock(spec=ServerClient),
    )


@pytest.mark.asyncio
async def test_real_seed_request_prevents_solver_on_missing_dependency(creator, monkeypatch, tmp_path):
    # Real seed endpoint, actual supervised subprocess in this missing-RDKit env.
    import importlib.util

    if importlib.util.find_spec("rdkit") is not None:
        pytest.skip("Run in isolated missing-RDKit controller environment")
    from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig, SimpleAgentRunRequest
    from responses_api_agents.simple_agent.tests.test_app import _mock_response

    monkeypatch.setattr(app.LocalProvider, "_semaphores", {})
    client = TestClient(resource(creator).setup_webserver())
    calls = []

    async def post(*, server_name, url_path, **kwargs):
        calls.append((server_name, url_path))
        assert url_path == "/seed_session"
        response = client.post(url_path, json=kwargs["json"])
        assert response.status_code == 503, response.text
        assert response.json()["detail"]["uuid"] == creator[1]
        reply = _mock_response(status=response.status_code, content=response.text)
        reply.raise_for_status.side_effect = ClientResponseError(
            request_info=SimpleNamespace(
                url="http://resource/seed_session", real_url="http://resource/seed_session", method="POST", headers={}
            ),
            history=(),
            status=503,
            message="preflight failed",
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
            resources_server={"type": "resources_servers", "name": "resource"},
            model_server={"type": "responses_api_models", "name": "policy_model"},
            max_steps=1,
        ),
        server_client=sc,
    )
    body = SimpleAgentRunRequest.model_validate(
        {
            "uuid": creator[1],
            "verifier_metadata": {},
            "responses_create_params": {"input": [{"role": "user", "content": "frozen input"}]},
            "_ng_task_index": 3,
            "_ng_rollout_index": 0,
        }
    )
    with pytest.raises(ClientResponseError) as error:
        await agent.run(MagicMock(cookies={}, path_params={}), body)
    assert b"dependency_preflight_failed" in error.value.response_content
    assert calls == [("resource", "/seed_session")]

    # The real collector persists the same native agent error outside scientific scores.
    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper, failures_path_for
    from tests.unit_tests.test_rollout_collection import install_fake_server_client

    calls.clear()

    async def dispatch(server_name, url_path, json, **kwargs):
        assert url_path == "/run"
        return await agent.run(MagicMock(cookies={}, path_params={}), SimpleAgentRunRequest.model_validate(json))

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=dispatch))
    monkeypatch.setattr("nemo_gym.rollout_collection.get_global_config_dict", lambda: {})
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    row = body.model_dump(mode="json") | {"agent_ref": {"name": "my_agent"}}
    input_path.write_text(json.dumps(row) + "\n")
    with pytest.raises(RuntimeError, match="run has no score to report"):
        await RolloutCollectionHelper().run_from_config(
            RolloutCollectionConfig(
                input_jsonl_fpath=str(input_path),
                output_jsonl_fpath=str(output_path),
                route_failures_to_sidecar=True,
                disable_health_check=True,
                disable_aggregation=True,
            )
        )
    failures = [json.loads(line) for line in failures_path_for(output_path).read_text().splitlines()]
    assert len(failures) == 1
    assert creator[1] in failures[0]["_ng_failure_response_body"]
    assert failures[0]["_ng_task_index"] == 3
    assert "dependency_preflight_failed" in failures[0]["_ng_failure_response_body"]
    assert "reward" not in failures[0]
    assert not output_path.read_text().strip()
    assert calls == [("resource", "/seed_session")]


def test_modern_seed_identity_and_uncovered_branch(creator, monkeypatch):
    monkeypatch.setattr(app, "run_score", AsyncMock(side_effect=RuntimeError("missing dependency")))
    envelope = {
        "resources_session_id": "session",
        "episode_id": {"rollout_id": "rollout"},
        "task_id": {"taskset": "frozen", "task_id": "task-3"},
        "task_data": {"uuid": creator[1]},
    }
    response = TestClient(resource(creator).setup_webserver()).post("/seed_session", json=envelope)
    assert response.status_code == 503
    assert response.json()["detail"]["task_id"] == envelope["task_id"]
    assert response.json()["detail"]["dispatch_status"] == "not_sent"
    response = TestClient(resource(creator, "fgbench").setup_webserver()).post("/seed_session", json=envelope)
    assert response.status_code == 200
    assert response.headers["x-nemo-gym-scorer-preflight"] == "not_covered"
    assert response.json()["resources_session_id"] == "session"


def test_seed_success_preserves_parent_envelopes(creator, monkeypatch):
    runner = AsyncMock(return_value={"status": "ready", "uuid": creator[1]})
    monkeypatch.setattr(app, "run_score", runner)
    client = TestClient(resource(creator).setup_webserver())
    legacy = client.post("/seed_session", json={"uuid": creator[1], "_ng_task_index": 9})
    assert legacy.status_code == 200
    assert legacy.json() == {}
    assert legacy.headers["x-nemo-gym-scorer-preflight"] == "chemiq_initialization_ready"
    modern = client.post(
        "/seed_session",
        json={
            "resources_session_id": "original-session",
            "episode_id": {"rollout_id": "original-rollout"},
            "task_id": {"taskset": "original", "task_id": "original-task"},
            "task_data": {"uuid": creator[1]},
        },
    )
    assert modern.status_code == 200
    from nemo_gym.base_resources_server import ResourcesSeedSessionResponse

    assert modern.json() == ResourcesSeedSessionResponse(resources_session_id="original-session").model_dump(
        mode="json"
    )
    assert runner.call_args.kwargs["preflight"] is True
    assert runner.call_args.kwargs["timeout_s"] == 60


def test_actual_worker_rejects_missing_uuid(creator, tmp_path):
    python = Path(
        os.environ.get("CHEMIQ_QUALIFIED_PYTHON", ROOT / "resources_servers/campaign_creator_scoring/.venv/bin/python")
    )
    if not python.exists():
        pytest.skip("Provide CHEMIQ_QUALIFIED_PYTHON")
    result, output = worker(python, (creator[0], "absent-frozen-uuid"), tmp_path)
    assert result.returncode != 0
    assert "Selected ChemIQ UUID is absent" in result.stderr
    assert not output.exists()
