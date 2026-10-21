# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.ctbench.grading import creator, parse_and_score
from resources_servers.superchem.tests.test_native_contracts import response


@pytest.fixture
def funcs():
    return creator(os.environ["CTBENCH_SOURCE"])


@pytest.mark.parametrize(
    "pairs,refs,cands,expected",
    [([["Age", "Age"]], [], [], 1), ([], ["Age"], ["Wrong"], 0), ([["Age", "Age"]], ["Sex"], [], 2 / 3)],
)
def test_scores(funcs, pairs, refs, cands, expected):
    raw = json.dumps(
        {"matched_features": pairs, "remaining_reference_features": refs, "remaining_candidate_features": cands}
    )
    scores = parse_and_score(raw, [p[0] for p in pairs] + refs, [p[1] for p in pairs] + cands, funcs)
    assert scores["f1"] == expected


def test_empty_creator_bug(funcs):
    raw = json.dumps(
        {"matched_features": [], "remaining_reference_features": ["Age"], "remaining_candidate_features": []}
    )
    with pytest.raises(ZeroDivisionError):
        parse_and_score(raw, ["Age"], [], funcs)


def test_hallucinated_features(funcs):
    raw = json.dumps(
        {"matched_features": [["Age", "Age"]], "remaining_reference_features": [], "remaining_candidate_features": []}
    )
    with pytest.raises(ValueError):
        parse_and_score(raw, ["Age"], ["Wrong"], funcs)


def test_native(monkeypatch):
    from resources_servers.ctbench import app

    judged = json.dumps(
        {"matched_features": [["Age", "Age"]], "remaining_reference_features": [], "remaining_candidate_features": []}
    )
    monkeypatch.setattr(app, "call_judge", AsyncMock(return_value=NeMoGymResponse(**response(judged))))
    config = app.Config(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="ctbench",
        domain="knowledge",
        creator_root=os.environ["CTBENCH_SOURCE"],
        judge_model_server={"type": "responses_api_models", "name": "policy_model"},
    )
    server = app.Server(config=config, server_client=MagicMock(spec=ServerClient))
    result = TestClient(server.setup_webserver()).post(
        "/verify",
        json={
            "responses_create_params": {"input": [{"role": "user", "content": "List baseline features"}]},
            "response": response("`Age`"),
            "verifier_metadata": {"reference": "`Age`", "trial_context": "Trial"},
        },
    )
    assert result.status_code == 200, result.text
    assert result.json()["reward"] == 1


def test_explicit_judge_budget_variant():
    from resources_servers.ctbench.app import Config

    kwargs = dict(
        host="127.0.0.1", port=8080, entrypoint="app.py", name="ctbench", domain="knowledge", creator_root="/unused"
    )
    assert Config(**kwargs).judge_max_output_tokens == 1000
    assert Config(**kwargs, judge_max_output_tokens=4096).judge_max_output_tokens == 4096
    with pytest.raises(ValueError):
        Config(**kwargs, judge_max_output_tokens=0)


def test_guarded_judge_identity_and_missing_lineage(tmp_path):
    import hashlib
    import shutil

    import httpx
    from fastapi import FastAPI
    from fastapi import Request as HttpRequest

    from resources_servers.ctbench import app

    source = tmp_path / "creator"
    shutil.copytree(os.environ["CTBENCH_SOURCE"], source)
    seen = []
    judge = FastAPI()

    @judge.post("/v1/responses")
    async def fixture(request: HttpRequest):
        seen.append((dict(request.headers), await request.json()))
        return response(
            json.dumps(
                {
                    "matched_features": [["Age", "Age"]],
                    "remaining_reference_features": [],
                    "remaining_candidate_features": [],
                }
            )
        )

    class Client:
        global_config_dict = {"provider_dispatch_guard": True}

        async def post(self, *, server_name, url_path, json, headers):
            assert server_name == "ctbench_judge"
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=judge), base_url="http://fixture"
            ) as client:
                reply = await client.post(url_path, json=json.model_dump(mode="json"), headers=headers)
            result = MagicMock()
            result.ok = reply.is_success
            result.status = reply.status_code
            result.read = AsyncMock(return_value=reply.content)
            result.content.read = AsyncMock(return_value=reply.content)
            return result

    config = app.Config(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="ctbench",
        domain="knowledge",
        creator_root=str(source),
        judge_model_server={"type": "responses_api_models", "name": "ctbench_judge"},
    )
    transport = Client()
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = transport.global_config_dict
    server_client.post = transport.post
    server = app.Server(config=config, server_client=server_client)
    client = TestClient(server.setup_webserver())
    body = {
        "responses_create_params": {"input": [{"role": "user", "content": "List baseline features"}]},
        "response": response("`Age`"),
        "verifier_metadata": {"reference": "`Age`", "trial_context": "Trial"},
        "task_id": "trial-1",
        "rollout_index": 2,
        "verification_id": "persisted-verification",
    }
    for _ in range(2):
        result = client.post("/verify", json=body)
        assert result.status_code == 200, result.text
        assert result.json()["reward"] == 1
        assert result.json()["task_id"] == "trial-1"
        assert result.json()["rollout_index"] == 2
    assert seen[0] == seen[1]
    assert seen[0][0]["x-nemo-gym-logical-call-id"] == "judge:persisted-verification:feature-matching"
    functions = creator(str(source))
    system, query = functions["build_gpt4_eval_prompt"](["Age"], ["Age"], "Trial")
    assert seen[0][1]["input"][0]["content"] == system
    assert seen[0][1]["input"][1]["content"] == query
    client.post("/verify", json={**body, "verification_id": "fresh-verification"})
    assert seen[-1][0]["x-nemo-gym-logical-call-id"] != seen[0][0]["x-nemo-gym-logical-call-id"]
    assert seen[-1][0]["x-nemo-gym-call-config-sha256"] == seen[0][0]["x-nemo-gym-call-config-sha256"]
    module = source / "module.py"
    module.write_bytes(module.read_bytes() + b"\n# source identity control\n")
    assert creator(str(source))["__creator_source_sha256"] == hashlib.sha256(module.read_bytes()).hexdigest()
    client.post("/verify", json=body)
    assert seen[-1][0]["x-nemo-gym-call-config-sha256"] != seen[0][0]["x-nemo-gym-call-config-sha256"]
    previous = len(seen)
    result = client.post("/verify", json={**body, "verification_id": None})
    assert result.status_code == 200
    assert result.json()["mask_sample"] is True
    assert result.json()["_ng_failure_class"] == "judge_failed"
    assert result.json()["task_id"] == "trial-1"
    assert len(seen) == previous
