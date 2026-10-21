# SPDX-License-Identifier: Apache-2.0
"""Real collector/provider HTTP composition; no external inference."""

import asyncio
import time
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientResponseError, ClientSession, web
from omegaconf import OmegaConf

from nemo_gym._checkpoint.control import ControlError
from nemo_gym.base_responses_api_model import CaptureStore
from nemo_gym.provider_dispatch import ProviderDispatch
from nemo_gym.rollout_collection import RolloutCollectionHelper
from nemo_gym.server_utils import ServerClient
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_after", ["0.12", None])
@pytest.mark.parametrize("drain", [False, True])
async def test_rejection_preserves_peer_and_delays_queue(monkeypatch, retry_after, drain):
    model = SimpleModelServer(
        config=SimpleModelServerConfig(
            host="127.0.0.1",
            port=1,
            name="policy",
            entrypoint="",
            openai_base_url="http://unused",
            openai_api_key="fake",
            openai_model="fixture",
            max_concurrent_requests=2,
            upstream_max_num_tries=1,
            upstream_rate_limit_cooldown_seconds=0.12,
            upstream_retry_policy={"max_attempts": 1},
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    accepted = asyncio.Event()
    calls = []
    peer_finished = False

    async def provider(request):
        nonlocal peer_finished
        index = (await request.json())["index"]
        calls.append((index, time.monotonic()))
        if index == 0:
            await accepted.wait()
            return web.Response(status=429, headers={"Retry-After": retry_after} if retry_after else {})
        if index == 1:
            accepted.set()
            await asyncio.sleep(0.04)
            peer_finished = True
        return web.json_response({"reward": 0, "failure_kind": "model_budget" if index == 1 else None})

    app = web.Application()
    app.router.add_post("/provider", provider)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    try:
        async with ClientSession() as session:

            async def run(request):
                row = await request.json()

                async def operation():
                    async with session.post(url + "/provider", json={"index": row["_ng_task_index"]}) as response:
                        response.raise_for_status()
                        return await response.json()

                try:
                    return web.json_response(await model._call_upstream(operation))
                except ClientResponseError:
                    return web.json_response({"detail": "upstream rejected"}, status=500)

            agent = web.Application()
            agent.router.add_post("/run", run)
            agent_runner = web.AppRunner(agent)
            await agent_runner.setup()
            agent_site = web.TCPSite(agent_runner, "127.0.0.1", 0)
            await agent_site.start()
            agent_url = f"http://127.0.0.1:{agent_site._server.sockets[0].getsockname()[1]}"
            client = MagicMock()
            client.global_config_dict = OmegaConf.create(
                {
                    "agent": {"responses_api_agents": {"impl": {}}},
                    "environment": {"environment_servers": {"legacy_agent": {"agent_server": {"name": "agent"}}}},
                }
            )

            async def post(*, server_name, url_path, json):
                return await session.post(agent_url + url_path, json=json)

            client.post = post
            monkeypatch.setattr("nemo_gym.rollout_collection.setup_server_client_utils", lambda *a, **kw: client)
            rows = [
                {
                    "agent_ref": {"name": "agent"},
                    "_ng_task_index": i,
                    "_ng_rollout_index": 0,
                    "responses_create_params": {"input": []},
                }
                for i in range(5 if drain else 3)
            ]
            iterator = RolloutCollectionHelper().run_examples(
                rows,
                semaphore=asyncio.Semaphore(2),
                route_failures_to_sidecar=True,
                dispatch_budget_s=0.08 if drain else 1,
                drain_margin_s=0.02 if drain else 0.1,
            )
            try:
                result = {row["_ng_task_index"]: value for row, value in [await future for future in iterator]}
            finally:
                await iterator.aclose()
                await agent_runner.cleanup()
            assert peer_finished
            assert "reward" not in result[0] and "response" not in result[0]
            assert result[0]["_ng_failure_http_status"] == 500
            assert result[1]["failure_kind"] == "model_budget" and result[1]["reward"] == 0
            assert sorted(i for i, _ in calls) == ([0, 1, 2, 3] if drain else [0, 1, 2])
            if drain:
                assert result[4]["_ng_dispatch_drained"] is True
                assert "reward" not in result[4]
            assert dict(calls)[2] - dict(calls)[0] >= 0.10
    finally:
        await runner.cleanup()


def test_unknown_capture_still_refuses_restore(tmp_path):
    store = CaptureStore(tmp_path, provider_dispatch_guard=True)
    store.mark_pending("episode", "call")
    dispatch = ProviderDispatch(lambda evidence: store.write_dispatch("episode", "call", evidence))
    dispatch.before_send()
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ControlError, match="unresolved provider"):
        store.assert_restore_safe(["episode"])
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
