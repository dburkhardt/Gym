# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harbor call identity survives explicit checkpoint recovery, never fresh trials."""

import json
import shutil
from unittest.mock import AsyncMock

import httpx
import pytest

from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import NemoGymLLM


def make_llm(path, *, resume=False):
    return NemoGymLLM(
        model_name="test",
        api_base="http://localhost/ng-rollout/episode/v1",
        dispatch_state_path=path,
        dispatch_resume_preserved_state=resume,
    )


@pytest.mark.asyncio
async def test_binding_survives_checkpoint_move_and_rejects_competing_owner(tmp_path):
    path = tmp_path / "trial" / "agent" / "gym-call-state.json"
    original = make_llm(path)
    payload = {"model": "test", "messages": [{"role": "user", "content": "hello"}]}
    first = original._dispatch_headers(payload)
    with pytest.raises(RuntimeError, match="Another Harbor client"):
        make_llm(path, resume=True)._dispatch_headers(payload)
    assert original._dispatch_headers(payload) == first
    await original.aclose()
    with pytest.raises(RuntimeError, match="preserved environment/chat"):
        make_llm(path)._dispatch_headers(payload)
    moved = tmp_path / "restored" / "agent" / "gym-call-state.json"
    moved.parent.mkdir(parents=True)
    shutil.copyfile(path, moved)
    restored = make_llm(moved, resume=True)
    assert restored._dispatch_headers(payload) == first
    with pytest.raises(RuntimeError, match="binding differs"):
        restored._dispatch_headers({**payload, "temperature": 0.5})
    await restored.aclose()
    fresh = make_llm(tmp_path / "fresh" / "agent" / "gym-call-state.json")
    assert fresh._dispatch_headers(payload)["x-nemo-gym-logical-call-id"] != first["x-nemo-gym-logical-call-id"]
    await fresh.aclose()


@pytest.mark.asyncio
async def test_retry_and_completed_checkpoint_use_native_chat_cursor(tmp_path):
    path = tmp_path / "trial" / "agent" / "gym-call-state.json"
    llm = make_llm(path)
    response = httpx.Response(200, json={"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]})
    client = AsyncMock()
    client.post.side_effect = [TimeoutError("unknown provider outcome"), response]
    llm._http_client = client
    with pytest.raises(TimeoutError):
        await NemoGymLLM.call.__wrapped__(llm, "question")
    assert not json.loads(path.read_text())["calls"]["0"]["completed"]
    await NemoGymLLM.call.__wrapped__(llm, "question")
    assert json.loads(path.read_text())["calls"]["0"]["completed"]
    first_id = client.post.call_args_list[0].kwargs["headers"]["x-nemo-gym-logical-call-id"]
    assert client.post.call_args_list[1].kwargs["headers"]["x-nemo-gym-logical-call-id"] == first_id
    await llm.aclose()
    restored = make_llm(path, resume=True)
    restored._http_client = AsyncMock()
    restored._http_client.post.return_value = response
    # Same persisted turn replays the completed capture, rather than generating again.
    await NemoGymLLM.call.__wrapped__(restored, "question")
    assert restored._http_client.post.call_args.kwargs["headers"]["x-nemo-gym-logical-call-id"] == first_id
    history = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]
    await NemoGymLLM.call.__wrapped__(restored, "next question", history)
    assert restored._http_client.post.call_args.kwargs["headers"]["x-nemo-gym-logical-call-id"] != first_id
    await restored.aclose()


def test_guard_requires_capture_route(tmp_path):
    llm = NemoGymLLM(model_name="test", api_base="http://localhost/v1", dispatch_state_path=tmp_path / "state.json")
    with pytest.raises(RuntimeError, match="rollout-qualified"):
        llm._dispatch_headers({"model": "test"})


@pytest.mark.asyncio
async def test_corrupt_state_never_becomes_fresh_on_retry(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{broken")
    llm = make_llm(path, resume=True)
    for _ in range(2):
        with pytest.raises(json.JSONDecodeError):
            await NemoGymLLM.call.__wrapped__(llm, "question")
        assert llm._dispatch_lock is None
    assert path.read_text() == "{broken"


@pytest.mark.asyncio
async def test_overlapping_calls_are_rejected(tmp_path):
    import asyncio

    from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import ConcurrentHarborCallError

    llm = make_llm(tmp_path / "state.json")
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await release.wait()
        raise TimeoutError("control")

    llm._post_chat_completions = blocked
    first = asyncio.create_task(NemoGymLLM.call.__wrapped__(llm, "question"))
    await entered.wait()
    with pytest.raises(ConcurrentHarborCallError):
        await NemoGymLLM.call.__wrapped__(llm, "other question")
    release.set()
    with pytest.raises(TimeoutError):
        await first
    await llm.aclose()


@pytest.mark.asyncio
async def test_native_capture_replays_completed_call_after_client_restart(tmp_path):
    from nemo_gym.base_responses_api_model import CaptureStore, _CaptureMiddleware
    from nemo_gym.provider_dispatch import current_provider_dispatch

    physical = []

    async def provider(scope, receive, send):
        await receive()
        dispatch = current_provider_dispatch.get()
        dispatch.before_send()
        physical.append("sent")
        dispatch.response_headers(200)
        body = {"id": "control", "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]}
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": json.dumps(body).encode()})

    state = tmp_path / "agent" / "state.json"
    for resume in (False, True):
        capture = _CaptureMiddleware(
            provider,
            store=CaptureStore(tmp_path / "capture", provider_dispatch_guard=True),
            model_server_name="solver",
            model_config_digest="b" * 64,
        )
        llm = make_llm(state, resume=resume)
        llm._http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=capture))
        result = await NemoGymLLM.call.__wrapped__(llm, "question")
        assert result.content == "answer"
        await llm.aclose()
    assert physical == ["sent"]


@pytest.mark.asyncio
async def test_truncated_response_is_durable_before_chat_advances(tmp_path):
    from harbor.llms.base import OutputLengthExceededError

    path = tmp_path / "state.json"
    llm = make_llm(path)
    llm._http_client = AsyncMock()
    llm._http_client.post.return_value = httpx.Response(200, json={
        "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]
    })
    with pytest.raises(OutputLengthExceededError) as error:
        await NemoGymLLM.call.__wrapped__(llm, "question")
    assert error.value.truncated_response == "partial"
    assert json.loads(path.read_text())["calls"]["0"]["completed"]
    first = llm._http_client.post.call_args.kwargs["headers"]["x-nemo-gym-logical-call-id"]
    await llm.aclose()
    restored = make_llm(path, resume=True)
    restored._http_client = AsyncMock()
    restored._http_client.post.return_value = httpx.Response(200, json={
        "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]
    })
    with pytest.raises(OutputLengthExceededError) as replay_error:
        await NemoGymLLM.call.__wrapped__(restored, "question")
    assert replay_error.value.truncated_response == "partial"
    assert restored._http_client.post.call_args.kwargs["headers"]["x-nemo-gym-logical-call-id"] == first
    restored._http_client.post.return_value = httpx.Response(200, json={
        "choices": [{"message": {"content": "finished"}, "finish_reason": "stop"}]
    })
    history = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "partial"}]
    result = await NemoGymLLM.call.__wrapped__(restored, "continue", history)
    assert result.content == "finished"
    assert restored._http_client.post.call_args.kwargs["headers"]["x-nemo-gym-logical-call-id"] != first
    assert len(json.loads(path.read_text())["calls"]) == 2
    await restored.aclose()
