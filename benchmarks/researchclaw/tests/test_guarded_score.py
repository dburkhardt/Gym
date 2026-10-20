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


@pytest.mark.asyncio
async def test_cancel_settles_creator_before_restoring_globals(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "_meta.json").write_text(json.dumps({"task_id": "Life_002"}))
    refs = tmp_path / "tasks/Life_002/target_study"
    refs.mkdir(parents=True)
    (refs / "checklist.json").write_text(json.dumps([{"weight": 1}]))
    entered, release = threading.Event(), threading.Event()
    original_agent, original_threads = object(), object()
    creator = SimpleNamespace(
        __file__=str(tmp_path / "evaluation/score.py"), LLMAgent=original_agent, multi_thread=original_threads
    )

    def native(workspace):
        entered.set()
        release.wait(5)
        assert creator.LLMAgent is not original_agent
        return {"items": []}

    creator.score_workspace = native
    monkeypatch.setattr(scoring, "SOURCE_HASHES", {})
    monkeypatch.setattr(scoring.importlib.metadata, "version", lambda name: "0.1.23")
    monkeypatch.setattr(scoring.importlib, "import_module", lambda name: creator)
    call = asyncio.create_task(
        scoring.score(tmp_path, workspace, "http://localhost/ng-rollout/test/v1", "lineage", "Life_002")
    )
    await asyncio.to_thread(entered.wait)
    call.cancel()
    await asyncio.sleep(0.01)
    call.cancel()
    assert not call.done()
    assert creator.LLMAgent is not original_agent
    with pytest.raises(RuntimeError, match="already owns"):
        await scoring.score(tmp_path, workspace, "http://localhost/ng-rollout/test/v1", "other", "Life_002")
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert creator.LLMAgent is original_agent and creator.multi_thread is original_threads
    assert not scoring._SCORER_OWNER.locked()


@pytest.mark.asyncio
async def test_stopped_transport_never_dispatches():
    import threading

    stop = threading.Event()
    stop.set()
    calls = []

    async def send(*args):
        calls.append(args)

    transport = scoring.CriterionTransport(send, asyncio.get_running_loop(), "lineage", 0, "digest", stop=stop)
    with pytest.raises(RuntimeError, match="unresolved"):
        await asyncio.to_thread(transport.create, model=scoring.MODEL)
    assert not calls and transport.settled.is_set()


@pytest.mark.asyncio
async def test_http_body_read_is_bounded_and_released(monkeypatch):
    released = []

    class Response:
        def raise_for_status(self):
            pass

        async def json(self):
            await asyncio.Event().wait()

        def release(self):
            released.append(True)

    async def request(*args, **kwargs):
        assert kwargs["_max_num_tries"] == 1 and not kwargs["_resend_after_disconnect"]
        return Response()

    monkeypatch.setattr(scoring, "request", request)
    monkeypatch.setattr(scoring, "CREATOR_REQUEST_TIMEOUT_S", 0.01)
    with pytest.raises(TimeoutError):
        await scoring._send_guarded("http://localhost/ng-rollout/test/v1", {"model": scoring.MODEL}, {})
    assert released == [True]


@pytest.mark.asyncio
async def test_cancel_during_pending_transport_cleanup_restores_state(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "_meta.json").write_text(json.dumps({"task_id": "Life_002"}))
    refs = tmp_path / "tasks/Life_002/target_study"
    refs.mkdir(parents=True)
    (refs / "checklist.json").write_text(json.dumps([{"weight": 1}]))
    entered = threading.Event()
    release = asyncio.Event()
    worker_returned = threading.Event()
    threads = []

    class Response:
        def raise_for_status(self):
            pass

        async def json(self):
            return completion('{"score":50}').model_dump()

        def release(self):
            pass

    async def request(*args, **kwargs):
        entered.set()
        await release.wait()
        return Response()

    class Agent:
        def __init__(self, **kwargs):
            self.client = SimpleNamespace(close=lambda: None)

        def __call__(self, *args, **kwargs):
            thread = threading.Thread(target=lambda: self.client.chat.completions.create(model=scoring.MODEL))
            threads.append(thread)
            thread.start()
            entered.wait(5)
            return {"score": 50}  # Mimics creator timeout returning while API thread is active.

    def multi(inputs, function, **kwargs):
        return [function(i, item) for i, item in enumerate(inputs)]

    creator = SimpleNamespace(__file__=str(tmp_path / "evaluation/score.py"), LLMAgent=Agent, multi_thread=multi)

    def native(workspace):
        try:
            creator.multi_thread([{}], lambda **kwargs: creator.LLMAgent()())
        except RuntimeError:
            pass
        worker_returned.set()
        return {"items": []}

    creator.score_workspace = native
    monkeypatch.setattr(scoring, "SOURCE_HASHES", {})
    monkeypatch.setattr(scoring.importlib.metadata, "version", lambda name: "0.1.23")
    monkeypatch.setattr(scoring.importlib, "import_module", lambda name: creator)
    monkeypatch.setattr(scoring, "request", request)
    monkeypatch.setenv("JUDGE_API_BASE", "original-base")
    before = list(sys.path)
    call = asyncio.create_task(
        scoring.score(tmp_path, workspace, "http://localhost/ng-rollout/test/v1", "id", "Life_002")
    )
    await asyncio.to_thread(worker_returned.wait)
    call.cancel()
    await asyncio.sleep(0.01)
    assert not call.done() and creator.LLMAgent is not Agent
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await call
    for thread in threads:
        thread.join(1)
        assert not thread.is_alive()
    assert creator.LLMAgent is Agent and creator.multi_thread is multi
    assert os.environ["JUDGE_API_BASE"] == "original-base" and sys.path == before
