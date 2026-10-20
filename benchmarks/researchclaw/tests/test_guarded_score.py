# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from openai.types.chat import ChatCompletion

from benchmarks.researchclaw import guarded_score as scoring


def completion(content):
    return ChatCompletion.model_validate(
        {
            "id": "fake-response",
            "object": "chat.completion",
            "created": 0,
            "model": scoring.MODEL,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        }
    )


@pytest.mark.asyncio
async def test_pending_transport_cannot_be_retried():
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def send(payload, headers):
        calls.append(headers)
        entered.set()
        await release.wait()
        return completion('{"score": 50}')

    transport = scoring.CriterionTransport(send, asyncio.get_running_loop(), "lineage", 3, "digest")
    first = asyncio.create_task(asyncio.to_thread(transport.create, model=scoring.MODEL))
    await entered.wait()
    with pytest.raises(RuntimeError, match="unresolved"):
        await asyncio.to_thread(transport.create, model=scoring.MODEL)
    release.set()
    await first
    assert len(calls) == 1
    assert calls[0]["x-nemo-gym-logical-call-id"] == "researchclaw:lineage:criterion:3:try:0"


@pytest.mark.asyncio
async def test_transport_failure_poisoned_not_parse_retry():
    calls = []

    async def send(payload, headers):
        calls.append(headers)
        raise TimeoutError("accepted but response unavailable")

    transport = scoring.CriterionTransport(send, asyncio.get_running_loop(), "lineage", 0, "digest")
    with pytest.raises(TimeoutError):
        await asyncio.to_thread(transport.create, model=scoring.MODEL)
    with pytest.raises(RuntimeError, match="unresolved"):
        await asyncio.to_thread(transport.create, model=scoring.MODEL)
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_transport", [False, True])
async def test_actual_pinned_creator_weighting_and_parser(tmp_path, monkeypatch, bad_transport):
    source_env = os.environ.get("RESEARCHCLAW_PINNED_SOURCE")
    if not source_env:
        pytest.skip("Stage pinned creator source and structai0.1.23 for offline creator test")
    source = tmp_path / "source"
    shutil.copytree(Path(source_env) / "evaluation", source / "evaluation")
    refs = source / "tasks" / "Life_002" / "target_study"
    refs.mkdir(parents=True)
    (refs / "checklist.json").write_text(
        json.dumps([{"content": "criterion-a", "weight": 1}, {"content": "criterion-b", "weight": 3}])
    )
    workspace = tmp_path / "workspace"
    (workspace / "report").mkdir(parents=True)
    (workspace / "report/report.md").write_text("Control report")
    (workspace / "INSTRUCTIONS.md").write_text("Frozen instructions")
    (workspace / "_meta.json").write_text(json.dumps({"task_id": "Life_002", "run_id": "test"}))
    for key in list(sys.modules):
        if key == "evaluation" or key.startswith("evaluation."):
            monkeypatch.delitem(sys.modules, key)
    calls = []

    class Response:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        async def json(self):
            return self.body

        def release(self):
            pass

    async def request(method, url, **kwargs):
        header = kwargs["headers"]["x-nemo-gym-logical-call-id"]
        calls.append((header, kwargs["json"]))
        if bad_transport:
            raise TimeoutError("ambiguous provider")
        index = int(header.split(":criterion:")[1].split(":")[0])
        attempt = int(header.rsplit(":", 1)[1])
        # Real structai parsing rejects this first response and exercises creator max_try2.
        content = "{}" if attempt == 0 else json.dumps({"score": 0 if index == 0 else 100, "reasoning": "control"})
        return Response(completion(content).model_dump())

    monkeypatch.setattr(scoring, "request", request)
    result = await scoring.score(source, workspace, "http://localhost/ng-rollout/task/v1", "stable", "Life_002")
    assert all(
        payload["temperature"] == 0 and payload.get("max_tokens", payload.get("max_completion_tokens")) == 500
        for _, payload in calls
    )
    if bad_transport:
        assert result["reward"] is None and len(calls) == 2
    else:
        assert result["native"]["total_score"] == 75
        assert result["reward"] == 0.75 and not result["failures"]
        assert len(calls) == 4 and len({key for key, _ in calls}) == 4
