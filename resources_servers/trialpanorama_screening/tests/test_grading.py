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


def test_qualification_preserves_raw_and_model_failure_denominator():
    from nemo_gym.result_disposition import summarize_dispositions

    records = []
    for text in [
        "",
        '{"results":[{"PMID":"1","decision":"included"}]}',
        json.dumps({"results": [{"PMID": "1", "decision": "INCLUDED"}, {"PMID": "2", "decision": "EXCLUDED"}]}),
    ]:
        cfg = Config(
            host="127.0.0.1",
            port=8080,
            name="trialpanorama_screening",
            entrypoint="app.py",
            domain="knowledge",
            creator_root=os.environ["TRIALPANORAMA_SOURCE"],
        )
        server = Server(config=cfg, server_client=MagicMock(spec=ServerClient))
        reply = TestClient(server.setup_webserver()).post(
            "/verify",
            json={
                "responses_create_params": {"input": [{"role": "user", "content": "fixture"}]},
                "response": response(text),
                "verifier_metadata": {"sample": SAMPLE},
            },
        )
        assert reply.status_code == 200, reply.text
        records.append(reply.json())
    assert [r["reward"] for r in records] == [0.5, 1, 1]
    assert [r["mask_sample"] for r in records] == [True, True, False]
    summary = summarize_dispositions(records)
    assert summary["counts"]["completed_result_records"] == 3
    # All attempted outputs remain represented in the separately named strict metric.
    metrics = summary["per_metric"]
    assert metrics["complete_output_micro_accuracy_v1"]["valid"] == 3
    assert metrics["creator_micro_accuracy"]["invalid"] == 2
    assert metrics["complete_output_micro_accuracy_v1"]["reviewed_denominator"] == 3
    assert metrics["complete_output_micro_accuracy_v1"]["reviewed_eligible_scores"] == 3
    assert [r["metadata"]["verifier_result"]["rewards"]["complete_output_micro_accuracy_v1"] for r in records] == [
        0,
        0,
        1,
    ]


def test_duplicate_and_bad_confidence_not_usable():
    for items in [
        [{"PMID": "1", "decision": "included"}] * 2,
        [{"PMID": "1", "decision": "included", "confidence": "bad"}, {"PMID": "2", "decision": "excluded"}],
    ]:
        result = grade(json.dumps({"results": items}), creator_root=os.environ["TRIALPANORAMA_SOURCE"], sample=SAMPLE)
        assert result["mask_sample"]
        assert result["metadata"]["verifier_result"]["rewards"]["complete_output_micro_accuracy_v1"] == 0


def test_dirty_creator_source_refused(monkeypatch):
    from pathlib import Path

    import pytest

    from resources_servers.trialpanorama_screening import grading

    original = Path.read_bytes

    def changed(path):
        value = original(path)
        return value + b"\n# dirty" if path.name == "study_screening.py" else value

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError, match="working tree"):
        grading.creator(os.environ["TRIALPANORAMA_SOURCE"])


def test_reference_duplicates_are_not_model_failures():
    import copy

    for label in ["included", "excluded"]:
        sample = copy.deepcopy(SAMPLE)
        sample["labels"].append({"PMID": "1", "label": label})
        result = grade("", creator_root=os.environ["TRIALPANORAMA_SOURCE"], sample=sample)
        assert result["failure_kind"] == "trialpanorama:reference_duplicate"
        assert all(m["status"] == "unresolved" for m in result["metadata"]["measurement_review"]["metrics"].values())
