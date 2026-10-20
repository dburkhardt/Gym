# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from nemo_gym.openai_utils import NeMoGymResponse
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
