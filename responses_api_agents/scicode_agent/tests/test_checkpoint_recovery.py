import asyncio
import json
import time
from unittest.mock import patch

import pytest

from nemo_gym._checkpoint.agent import AgentSessionParticipant
from nemo_gym._checkpoint.control import CheckpointRequest, CommitRequest, ParticipantControlPlane, RestoreRequest
from nemo_gym.base_responses_api_model import CaptureStore, _CaptureMiddleware
from nemo_gym.episode_types import EpisodeId
from nemo_gym.provider_dispatch import current_provider_dispatch
from responses_api_agents.scicode_agent.tests.test_app import _agent, _FakeRequest, _model_json, _Resp, _run_request


@pytest.mark.asyncio
async def test_native_checkpoint_replays_pending_step_without_duplicate_provider(tmp_path):
    physical = []
    requests = []
    response_saved = asyncio.Event()
    never = asyncio.Event()
    capture_path = tmp_path / "capture"

    def attach(agent, pause):
        agent.server_client.global_config_dict["provider_dispatch_guard"] = True
        agent._checkpoint_participant = AgentSessionParticipant(agent)
        store = CaptureStore(capture_path, provider_dispatch_guard=True)

        async def provider(scope, receive, send):
            await receive()
            dispatch = current_provider_dispatch.get()
            dispatch.before_send()
            physical.append(scope["path"])
            dispatch.response_headers(200)
            await send(
                {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]}
            )
            await send({"type": "http.response.body", "body": json.dumps(_model_json("x = 1")).encode()})

        middleware = _CaptureMiddleware(
            provider, store=store, model_server_name="policy_model", model_config_digest="b" * 64
        )

        async def post(**kwargs):
            if kwargs["url_path"] == "/verify":
                return _Resp({"reward": 1, "solutions": kwargs["json"]["solutions"]})
            requests.append(kwargs)
            messages = [{"type": "http.request", "body": json.dumps(kwargs["json"]).encode()}]
            sent = []

            async def receive():
                return messages.pop(0)

            async def send(message):
                sent.append(message)

            await middleware(
                {
                    "type": "http",
                    "method": "POST",
                    "path": kwargs["url_path"],
                    "headers": [(k.encode(), v.encode()) for k, v in kwargs["headers"].items()],
                },
                receive,
                send,
            )
            if pause and len(requests) == 2:
                response_saved.set()
                await never.wait()
            return _Resp(json.loads(sent[-1]["body"]))

        agent.server_client.post.side_effect = post
        return ParticipantControlPlane(agent.checkpoint_participant, instance_name="agent", lease_grace_seconds=60)

    def ctl(checkpoint_id="c1", **kwargs):
        return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 10, **kwargs}

    original = _agent()
    controller = attach(original, True)
    body = _run_request(_ng_rollout_id="sci")
    with patch("responses_api_agents.scicode_agent.app.raise_for_status", return_value=None):
        task = asyncio.create_task(original.run(_FakeRequest(), body))
        await asyncio.wait_for(response_saved.wait(), 5)
        await controller.prepare(CheckpointRequest(**ctl()))
        await controller.commit(CommitRequest(**ctl(checkpoint_dir=str(tmp_path / "checkpoint"))))
        [record] = await original.checkpoint_participant.export(None)
        assert record.episode["next_step"] == 1
        assert len(record.episode["step_usage"]) == 1
        assert record.episode["generation_session_id"]
        task.cancel()
        await controller.resume(CheckpointRequest(**ctl()))
        with pytest.raises(asyncio.CancelledError):
            await task
        restarted = _agent()
        restored = attach(restarted, False)
        await restored.restore(
            RestoreRequest(
                **ctl(
                    "restore", checkpoint_dir=str(tmp_path / "checkpoint"), episode_ids=[EpisodeId(rollout_id="sci")]
                )
            )
        )
        await restored.resume(CheckpointRequest(**ctl("restore")))
        result = await restarted.run(_FakeRequest(), _run_request(_ng_rollout_id="sci-a1"))
    assert result["reward"] == 1
    assert len(physical) == 2
    assert len(requests) == 3
    assert requests[1]["headers"] == requests[2]["headers"]
    assert requests[1]["json"] == requests[2]["json"]
    assert "/sci-a1/" in requests[2]["url_path"]


class SavedBoundary:
    def __init__(self, continuation=None):
        self.continuation = continuation
        self.saved = None

    async def boundary(self, state):
        self.saved = state

    def step(self, mode):
        from contextlib import nullcontext

        return nullcontext()


@pytest.mark.asyncio
async def test_fresh_scientific_attempt_has_distinct_identity_and_checkpoint_precedes_call():
    seen = []
    for _ in range(2):
        agent = _agent(observability_enabled=True)
        agent.server_client.global_config_dict["provider_dispatch_guard"] = True
        state = SavedBoundary()

        async def stop(**kwargs):
            assert state.saved["generation_session_id"]
            assert state.saved["next_step"] == 0
            seen.append(kwargs["headers"]["x-nemo-gym-logical-call-id"])
            raise TimeoutError("accepted outcome unknown")

        agent.server_client.post.side_effect = stop
        with pytest.raises(TimeoutError):
            await agent._run(_FakeRequest(), _run_request(_ng_rollout_id="same-task"), legacy_run=state)
    assert seen[0] != seen[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage", ["missing_identity", "changed_scientific_config", "different_rollout", "unknown_stage"]
)
async def test_invalid_continuation_refuses_before_any_dispatch(damage):
    agent = _agent(observability_enabled=True)
    agent.server_client.global_config_dict["provider_dispatch_guard"] = True
    state = SavedBoundary()
    agent.server_client.post.side_effect = TimeoutError("stop")
    with pytest.raises(TimeoutError):
        await agent._run(_FakeRequest(), _run_request(_ng_rollout_id="sci"), legacy_run=state)
    saved = state.saved
    body = _run_request(_ng_rollout_id="sci-a1")
    if damage == "missing_identity":
        saved.pop("generation_session_id")
    elif damage == "changed_scientific_config":
        body.required_dependencies = "import pandas as pd"
    elif damage == "different_rollout":
        saved["capture_lineage"] = "other"
    else:
        saved["next"] = "unknown"
    agent.server_client.post.reset_mock()
    with pytest.raises(ValueError):
        await agent._run(_FakeRequest(), body, legacy_run=SavedBoundary(saved))
    agent.server_client.post.assert_not_called()
