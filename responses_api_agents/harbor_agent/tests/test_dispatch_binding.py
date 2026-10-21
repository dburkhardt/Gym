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
    llm._http_client.post.return_value = httpx.Response(
        200, json={"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}
    )
    with pytest.raises(OutputLengthExceededError) as error:
        await NemoGymLLM.call.__wrapped__(llm, "question")
    assert error.value.truncated_response == "partial"
    assert json.loads(path.read_text())["calls"]["0"]["completed"]
    first = llm._http_client.post.call_args.kwargs["headers"]["x-nemo-gym-logical-call-id"]
    await llm.aclose()
    restored = make_llm(path, resume=True)
    restored._http_client = AsyncMock()
    restored._http_client.post.return_value = httpx.Response(
        200, json={"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}
    )
    with pytest.raises(OutputLengthExceededError) as replay_error:
        await NemoGymLLM.call.__wrapped__(restored, "question")
    assert replay_error.value.truncated_response == "partial"
    assert restored._http_client.post.call_args.kwargs["headers"]["x-nemo-gym-logical-call-id"] == first
    restored._http_client.post.return_value = httpx.Response(
        200, json={"choices": [{"message": {"content": "finished"}, "finish_reason": "stop"}]}
    )
    history = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "partial"}]
    result = await NemoGymLLM.call.__wrapped__(restored, "continue", history)
    assert result.content == "finished"
    assert restored._http_client.post.call_args.kwargs["headers"]["x-nemo-gym-logical-call-id"] != first
    assert len(json.loads(path.read_text())["calls"]) == 2
    await restored.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["429", "timeout"])
async def test_guarded_outer_retry_does_not_reset_provider_attempt_budget(tmp_path, failure):
    llm = make_llm(tmp_path / "state.json")
    attempts = []

    def provider(request):
        attempts.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("ambiguous response", request=request)
        return httpx.Response(429, json={"error": "quota exhausted"})

    llm._http_client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    # Exercise the actual Tenacity-decorated public entry point, not __wrapped__.
    expected = httpx.ReadTimeout if failure == "timeout" else httpx.HTTPStatusError
    with pytest.raises(expected):
        await llm.call("question")
    assert len(attempts) == 1
    assert not json.loads((tmp_path / "state.json").read_text())["calls"]["0"]["completed"]
    await llm.aclose()


@pytest.mark.asyncio
async def test_guarded_successful_independent_turns_still_dispatch(tmp_path):
    llm = make_llm(tmp_path / "state.json")
    ids = []

    def provider(request):
        ids.append(request.headers["x-nemo-gym-logical-call-id"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]})

    llm._http_client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    await llm.call("question")
    await llm.call("next", [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}])
    assert len(ids) == len(set(ids)) == 2
    await llm.aclose()


@pytest.mark.asyncio
async def test_unguarded_legacy_retry_policy_remains_configurable():
    from tenacity import wait_none

    llm = NemoGymLLM(model_name="test", api_base="http://localhost/v1")
    calls = []

    def provider(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": "quota"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]})

    llm._http_client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    await NemoGymLLM.call.retry_with(wait=wait_none())(llm, "question")
    assert len(calls) == 2
    await llm.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("inner_attempts", [1, 2])
async def test_guarded_outer_submission_preserves_configured_inner_policy(tmp_path, inner_attempts):
    from unittest.mock import MagicMock

    from aiohttp import ClientResponseError

    from nemo_gym.server_utils import ServerClient
    from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig

    model = SimpleModelServer(
        config=SimpleModelServerConfig(
            host="127.0.0.1",
            port=8081,
            name="test",
            entrypoint="",
            openai_base_url="http://unused",
            openai_api_key="unused",
            openai_model="test",
            max_concurrent_requests=1,
            upstream_max_num_tries=1,
            upstream_rate_limit_cooldown_seconds=0,
            upstream_retry_policy={"max_attempts": inner_attempts, "backoff_initial_seconds": 0},
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    submissions, physical = [], []

    async def physical_operation():
        physical.append("send")
        if len(physical) == 1:
            raise ClientResponseError(None, (), status=429, headers={"Retry-After": "0"})
        return {"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]}

    async def model_route(request):
        submissions.append(request)
        try:
            body = await model._call_upstream(physical_operation)
            return httpx.Response(200, json=body)
        except ClientResponseError as error:
            return httpx.Response(error.status, json={"error": "quota"})

    llm = make_llm(tmp_path / "state.json")
    llm._http_client = httpx.AsyncClient(transport=httpx.MockTransport(model_route))
    try:
        if inner_attempts == 1:
            with pytest.raises(httpx.HTTPStatusError):
                await llm.call("question")
        else:
            assert (await llm.call("question")).content == "answer"
        assert len(submissions) == 1
        assert len(physical) == inner_attempts
    finally:
        await llm.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("image_variant", [False, True])
@pytest.mark.parametrize("failure", ["quota", "disconnect"])
async def test_guarded_terminus_query_does_not_repeat_model_submission(tmp_path, image_variant, failure):
    from unittest.mock import MagicMock
    from aiohttp import ClientResponseError
    from harbor.llms.chat import Chat
    from nemo_gym.server_utils import ServerClient
    from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig
    from responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym import Terminus2NemoGym
    from responses_api_agents.harbor_agent.custom_agents.terminus_2_images_nemo_gym import Terminus2ImagesNemoGym

    model = SimpleModelServer(
        config=SimpleModelServerConfig(
            host="127.0.0.1",
            port=8081,
            name="test",
            entrypoint="",
            openai_base_url="http://unused",
            openai_api_key="unused",
            openai_model="test",
            max_concurrent_requests=1,
            upstream_max_num_tries=1,
            upstream_rate_limit_cooldown_seconds=0,
            upstream_retry_policy={"max_attempts": 1, "backoff_initial_seconds": 0},
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    cls = Terminus2ImagesNemoGym if image_variant else Terminus2NemoGym
    agent = cls(
        logs_dir=tmp_path,
        model_name="test",
        api_base="http://localhost/ng-rollout/0-0/v1",
        gym_provider_dispatch_guard=True,
        enable_summarize=False,
        model_info={
            "max_input_tokens": 32000,
            "max_output_tokens": 4000,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        },
        **({"image_root": "/workspace/capsule"} if image_variant else {}),
    )
    submissions, physical = [], []

    async def operation():
        physical.append("send")
        if failure == "quota":
            raise ClientResponseError(None, (), status=429, headers={"Retry-After": "0"})
        raise TimeoutError("accepted request outcome unknown")

    async def route(request):
        submissions.append(request)
        try:
            await model._call_upstream(operation)
        except ClientResponseError:
            return httpx.Response(500, json={"error": "definitive provider rejection"})
        except TimeoutError:
            raise httpx.ReadTimeout("unknown", request=request)

    llm = agent._nemo_gym_llm
    llm._http_client = httpx.AsyncClient(transport=httpx.MockTransport(route))
    try:
        with pytest.raises((httpx.HTTPStatusError, httpx.ReadTimeout)):
            await agent._query_llm(Chat(llm), "original scientific instruction")
        assert len(submissions) == len(physical) == 1
        assert len(llm._dispatch_state["calls"]) == 1
    finally:
        await llm.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("guard, expected", [(True, 1), (False, 3)])
async def test_terminus_guard_retains_legacy_outer_retry_only_when_disabled(tmp_path, guard, expected):
    from harbor.llms.chat import Chat
    from responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym import Terminus2NemoGym

    agent = Terminus2NemoGym(
        logs_dir=tmp_path,
        model_name="test",
        api_base="http://localhost/ng-rollout/0-0/v1",
        gym_provider_dispatch_guard=guard,
        enable_summarize=False,
        model_info={
            "max_input_tokens": 32000,
            "max_output_tokens": 4000,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        },
    )
    requests = []

    def route(request):
        requests.append(request)
        return httpx.Response(500, json={"error": "failed"})

    llm = agent._nemo_gym_llm
    llm._http_client = httpx.AsyncClient(transport=httpx.MockTransport(route))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await agent._query_llm(Chat(llm), "original task")
        assert len(requests) == expected
    finally:
        await llm.aclose()
