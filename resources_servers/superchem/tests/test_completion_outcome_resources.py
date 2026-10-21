# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native endpoint diagnostics must not change any scorer's scientific values."""

import importlib
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient


CASES = [
    ("superchem", r"\boxed{A}", {"answer": ["A"], "question_type": "multiple_choice"}),
    ("moleculeqa", "B", {"answer": "B"}),
    ("reactbench", "CCO", {"answer": "CCO"}),
    ("microvqa_plusplus_demo", "2", {"correct_index": 2}),
]


def response(text, state):
    value = {
        "id": "fixture",
        "created_at": 0,
        "model": "fixture",
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
    if state != "completed":
        value["status"] = "incomplete"
        if state != "unspecified":
            value["incomplete_details"] = {"reason": "content_filter" if state == "unknown" else "max_output_tokens"}
        if state == "provider_error":
            value["error"] = {"code": "server_error", "message": "fixture provider failure"}
    return value


@pytest.mark.parametrize("name,text,metadata", CASES)
@pytest.mark.parametrize(
    "state,expected",
    [
        ("completed", None),
        ("budget", "model_budget"),
        ("unknown", "response:incomplete"),
        ("unspecified", "response:incomplete"),
        ("provider_error", "response:provider_error"),
    ],
)
@pytest.mark.parametrize("prior", [False, True])
def test_native_completion_diagnostics_preserve_scores_and_lineage(name, text, metadata, state, expected, prior):
    mod = importlib.import_module(f"resources_servers.{name}.app")
    cfg = mod.Config(host="127.0.0.1", port=8080, entrypoint="app.py", name=name, domain="knowledge")
    server = mod.Server(config=cfg, server_client=MagicMock(spec=ServerClient))
    payload = {
        "responses_create_params": {"input": [{"role": "user", "content": "fixture"}]},
        "response": response(text, state),
        "verifier_metadata": metadata,
        "task_id": "frozen-task",
        "uuid": "creator-id",
        "_ng_task_index": 7,
        "_ng_rollout_index": 2,
        "_ng_attempt_index": 1,
        "metadata": {"original": "retained"},
        "mask_sample": prior,
    }
    if prior:
        payload.update(failure_kind="existing:reference", failure_reason="preserved incident")
    scientific = mod.grade(text, **metadata)
    with TestClient(server.setup_webserver()) as client:
        result = client.post("/verify", json=payload)
    assert result.status_code == 200, result.text
    actual = result.json()
    assert actual["reward"] == scientific["reward"] == 1
    assert actual["diagnostics"] == scientific["diagnostics"]
    assert actual["mask_sample"] is prior
    assert actual.get("failure_kind") == ("existing:reference" if prior else expected)
    if prior:
        assert actual["failure_reason"] == "preserved incident"
    for key in ("task_id", "uuid", "_ng_task_index", "_ng_rollout_index", "_ng_attempt_index", "metadata"):
        assert actual[key] == payload[key]
    assert actual["response"]["status"] == payload["response"]["status"]
    if state == "provider_error":
        assert actual["response"]["error"] == payload["response"]["error"]


@pytest.mark.parametrize("name,text,metadata", CASES)
@pytest.mark.asyncio
@pytest.mark.parametrize("prior", [False, True])
async def test_empty_budget_zero_and_reason_only_incident(name, text, metadata, prior):
    mod = importlib.import_module(f"resources_servers.{name}.app")
    body = mod.Request.model_validate(
        {
            "responses_create_params": {"input": []},
            "response": response("", "budget"),
            "verifier_metadata": metadata,
            "failure_reason": "existing reason without kind" if prior else None,
            "mask_sample": False,
        }
    )
    result = await mod.Verifier().verify(body)
    assert result.reward == 0
    assert result.mask_sample is False
    assert result.failure_kind == (None if prior else "model_budget")
    if prior:
        assert result.failure_reason == "existing reason without kind"


@pytest.mark.parametrize(
    "name,relative",
    [
        ("superchem", "foundation/qa-native-controls/superchem-regraded20-v2.jsonl"),
        ("reactbench", "foundation/qa-native-controls/reactbench-regraded20.jsonl"),
        ("moleculeqa", "moleculeqa/pilot-20-v2.jsonl"),
        ("microvqa_plusplus_demo", "microvqa_plusplus_demo/pilot-15-combined.jsonl"),
    ],
)
@pytest.mark.asyncio
async def test_saved_outputs_preserve_exact_scoring_and_incidents(name, relative):
    import json
    import os
    from pathlib import Path

    root = os.environ.get("NEMO_GYM_COMPLETION_SAVED_ROOT")
    if not root:
        pytest.skip("Set NEMO_GYM_COMPLETION_SAVED_ROOT to preserved campaign records")
    path = Path(root) / relative
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert rows
    mod = importlib.import_module(f"resources_servers.{name}.app")
    for row in rows:
        body = mod.Request.model_validate(row)
        result = await mod.Verifier().verify(body)
        actual = result.model_dump()
        assert actual["reward"] == row["reward"]
        assert actual["diagnostics"] == row["diagnostics"]
        assert actual["mask_sample"] == row["mask_sample"]
        assert actual["response"] == body.response.model_dump()
        if row.get("failure_kind") or row.get("failure_reason"):
            assert actual["failure_kind"] == row.get("failure_kind")
            assert actual["failure_reason"] == row.get("failure_reason")
        for key in ("task_id", "task_source", "agent_ref", "_ng_task_index", "_ng_rollout_index", "_ng_attempt_index"):
            if key in row:
                assert actual[key] == row[key]
