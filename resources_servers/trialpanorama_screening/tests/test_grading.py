# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.superchem.tests.test_native_contracts import response
from resources_servers.trialpanorama_screening.app import Config, Server
from resources_servers.trialpanorama_screening.grading import grade


SAMPLE = {
    "inputs": {"criteria": "adult"},
    "context": [{"PMID": "1", "title": "x", "abstract": "y"}, {"PMID": "2", "title": "z", "abstract": "w"}],
    "labels": [{"PMID": "1", "label": "included"}, {"PMID": "2", "label": "excluded"}],
}


def test_right_wrong_empty():
    for pred, expected, valid in [
        ({"results": [{"PMID": "1", "decision": "included"}, {"PMID": "2", "decision": "excluded"}]}, 1, True),
        ({"results": [{"PMID": "1", "decision": "excluded"}, {"PMID": "2", "decision": "included"}]}, 0, True),
        ("", 0.5, False),
    ]:
        r = grade(json.dumps(pred) if pred else "", creator_root=os.environ["TRIALPANORAMA_SOURCE"], sample=SAMPLE)
        assert r["reward"] == expected
        assert r["diagnostics"]["strict_usable"] == valid


def test_creator_omission_inflation():
    r = grade(
        '{"results":[{"PMID":"1","decision":"included"}]}',
        creator_root=os.environ["TRIALPANORAMA_SOURCE"],
        sample=SAMPLE,
    )
    assert r["reward"] == 1
    assert r["diagnostics"]["complete_pmid_coverage"] is False


def test_native():
    cfg = Config(
        host="127.0.0.1",
        port=8080,
        name="trialpanorama_screening",
        entrypoint="app.py",
        domain="knowledge",
        creator_root=os.environ["TRIALPANORAMA_SOURCE"],
    )
    server = Server(config=cfg, server_client=MagicMock(spec=ServerClient))
    pred = json.dumps({"results": [{"PMID": "1", "decision": "included"}, {"PMID": "2", "decision": "excluded"}]})
    result = TestClient(server.setup_webserver()).post(
        "/verify",
        json={
            "responses_create_params": {"input": [{"role": "user", "content": "fixture"}]},
            "response": response(pred),
            "verifier_metadata": {"sample": SAMPLE},
        },
    )
    assert result.status_code == 200, result.text
    assert result.json()["reward"] == 1
