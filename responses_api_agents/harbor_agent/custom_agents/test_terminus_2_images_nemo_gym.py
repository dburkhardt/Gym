# SPDX-License-Identifier: Apache-2.0
import base64
import io
import json
import subprocess
import sys

import pytest
from PIL import Image

from responses_api_agents.harbor_agent.custom_agents.terminus_2_images_nemo_gym import ImageRegistry, reader_script


def png():
    stream = io.BytesIO()
    Image.new("RGB", (3, 2), "red").save(stream, format="PNG")
    return stream.getvalue()


@pytest.mark.parametrize("bad", ["../secret", "/secret", "a/../secret", "./secret"])
def test_bad_paths(bad):
    with pytest.raises(ValueError):
        reader_script("/workspace/results", bad, 100)


def test_reader_real_descriptors(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    (root / "image.png").write_bytes(png())

    def read(name, limit=1000):
        return subprocess.run([sys.executable, "-I", "-c", reader_script(str(root), name, limit)], capture_output=True)

    assert base64.b64decode(read("image.png").stdout) == png()
    (root / "link").symlink_to(root / "image.png")
    assert read("link").returncode != 0
    assert read("image.png", 1).returncode != 0
    (root / "directory").mkdir()
    assert read("directory").returncode != 0


def test_registry_exact_history_and_refusal(tmp_path):
    registry = ImageRegistry(tmp_path / "registry")
    token = registry.register(png())
    history = [{"role": "user", "content": token}, {"role": "assistant", "content": token}]
    projected = registry.project(history)
    assert projected[0]["content"][0]["text"] == history[0]["content"]
    assert base64.b64decode(projected[0]["content"][1]["image_url"]["url"].split(",")[1]) == png()
    assert projected[1] == history[1]
    assert registry.project(history) == projected
    with pytest.raises(ValueError, match="Missing"):
        ImageRegistry(tmp_path / "lost").project(history)
    with pytest.raises((ValueError, OSError)):
        registry.register(b"not an image")
    entry = next(iter(registry.entries.values()))
    from pathlib import Path

    Path(entry["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        registry.project(history)


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown,first_complete", [(False, False), (False, True), (True, False)])
async def test_guarded_converter_three_requests(tmp_path, monkeypatch, unknown, first_complete):
    import asyncio
    import socket
    from unittest.mock import MagicMock

    import uvicorn
    from aiohttp import ClientSession, web

    from nemo_gym import server_utils
    from nemo_gym.server_utils import ServerClient
    from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import NemoGymLLM
    from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig

    calls = []

    async def provider(request):
        calls.append(await request.json())
        if unknown:
            await asyncio.sleep(0.15)
        return web.json_response(
            {
                "id": "response-" + str(len(calls)),
                "created": 1,
                "model": "fixture-model",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "analysis": "view",
                                    "plan": "inspect",
                                    "commands": [{"keystrokes": "gym_view_image plot.png", "duration": "invalid"}]
                                    if len(calls) == 1
                                    else [],
                                    "task_complete": first_complete or len(calls) > 1,
                                }
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            }
        )

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", provider)
    runner = web.AppRunner(upstream)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    cfg = SimpleModelServerConfig(
        host="127.0.0.1",
        port=1,
        name="policy_model",
        entrypoint="app.py",
        openai_base_url=f"http://127.0.0.1:{port}/v1",
        openai_api_key="fixture",
        openai_model="fixture-model",
        upstream_max_num_tries=1,
        upstream_request_timeout_seconds=0.03 if unknown else 2,
    )
    model = SimpleModelServer(
        config=cfg,
        server_client=MagicMock(
            spec=ServerClient,
            global_config_dict={
                "observability_enabled": True,
                "provider_dispatch_guard": True,
                "model_call_capture_dir": str(tmp_path / "capture"),
            },
        ),
    )
    app = model.setup_webserver()
    model.setup_exception_middleware(app)
    model.setup_cancellation_middleware(app)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    host = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    serving = asyncio.create_task(host.serve(sockets=[listener]))
    while not host.started:
        await asyncio.sleep(0.001)
    client = NemoGymLLM(
        model_name="fixture-model",
        api_base=f"http://127.0.0.1:{listener.getsockname()[1]}/ng-rollout/0-0/v1",
        dispatch_state_path=tmp_path / "calls.json",
        responses_create_params={"input": [], "temperature": 0, "max_output_tokens": 128},
    )
    registry = ImageRegistry(tmp_path / "images")
    token = registry.register(png(), user_turn=0 if unknown else 1)
    client._project_messages = registry.project
    try:
        async with ClientSession() as physical:
            monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: physical)
            history = []
            if unknown:
                for _ in range(2):
                    with pytest.raises(Exception):
                        await client.call(token, history)
                assert len(calls) == 1
                for _ in range(100):
                    if list((tmp_path / "capture").glob("*.pending")):
                        break
                    await asyncio.sleep(0.01)
                pending = list((tmp_path / "capture").glob("*.pending"))
                assert len(pending) == 1
                assert json.loads(pending[0].read_text())["provider_attempts"][0]["outcome"] == "unknown"
            else:
                from types import SimpleNamespace

                from harbor.models.agent.context import AgentContext

                from responses_api_agents.harbor_agent.custom_agents.terminus_2_images_nemo_gym import (
                    Terminus2ImagesNemoGym,
                )

                class Session:
                    async def is_session_alive(self):
                        return True

                    async def get_incremental_output(self):
                        return "terminal ready"

                class Environment:
                    async def exec(self, command, timeout_sec):
                        return SimpleNamespace(return_code=0, stdout=base64.b64encode(png()).decode())

                agent = Terminus2ImagesNemoGym(
                    logs_dir=tmp_path / "native-agent",
                    model_name="fixture-model",
                    api_base=f"http://127.0.0.1:{listener.getsockname()[1]}/ng-rollout/0-0/v1",
                    gym_provider_dispatch_guard=True,
                    enable_summarize=False,
                    image_root="/workspace/results",
                    max_turns=3,
                    responses_create_params={"input": [], "temperature": 0, "max_output_tokens": 128},
                )
                agent._session = Session()
                await client.aclose()
                client = agent._nemo_gym_llm
                original = "Exact original scientific question."
                await agent.run(original, Environment(), AgentContext())
                assert original in calls[0]["messages"][0]["content"]
                assert "Harness tool:" in calls[0]["messages"][0]["content"]
                expected_calls = 2 if first_complete else 3
                assert agent._n_episodes == expected_calls
                image_text = calls[1]["messages"][-1]["content"][0]["text"]
                assert (
                    ("Previous response had warnings" in image_text)
                    if not first_complete
                    else ("Current terminal state" in image_text)
                )
                assert len(calls) == expected_calls
                serialized = [json.dumps(c) for c in calls]
                encoded = base64.b64encode(png()).decode()
                assert encoded not in serialized[0] and all(encoded in row for row in serialized[1:])
                assert all(
                    c["model"] == "fixture-model"
                    and c["temperature"] == 0
                    and c.get("max_tokens", c.get("max_completion_tokens")) == 128
                    for c in calls
                )
                for _ in range(100):
                    if (
                        sum(len(p.read_text().splitlines()) for p in (tmp_path / "capture").glob("*.capture.jsonl"))
                        == 3
                    ):
                        break
                    await asyncio.sleep(0.01)
                records = [
                    json.loads(line)
                    for p in (tmp_path / "capture").glob("*.capture.jsonl")
                    for line in p.read_text().splitlines()
                ]
                assert len(records) == expected_calls
                assert encoded in json.dumps(records[1])
    finally:
        await client.aclose()
        host.should_exit = True
        await serving
        listener.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_variant_hook_and_instruction_prefix(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from responses_api_agents.harbor_agent.custom_agents.terminus_2_images_nemo_gym import Terminus2ImagesNemoGym
    from responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym import Terminus2NemoGym

    seen = []

    async def parent_run(self, instruction, environment, context):
        seen.append(instruction)

    monkeypatch.setattr(Terminus2NemoGym, "run", parent_run)
    agent = Terminus2ImagesNemoGym(
        logs_dir=tmp_path / "logs",
        model_name="fixture",
        api_base="http://localhost:1/ng-rollout/0-0/v1",
        gym_provider_dispatch_guard=True,
        enable_summarize=False,
        image_root="/workspace/results",
    )

    class Environment:
        async def exec(self, command, timeout_sec):
            assert timeout_sec == 30
            assert "O_NOFOLLOW" in command
            return SimpleNamespace(return_code=0, stdout=base64.b64encode(png()).decode())

    original = "Exact frozen scientific question.\nDo not modify."
    await agent.run(original, Environment(), None)
    assert seen[0].startswith(original + "\nHarness tool:")
    assert seen[0][: len(original)] == original
    _, token = await agent._execute_commands([SimpleNamespace(keystrokes="gym_view_image plot.png")], None)
    messages = [
        {"role": "user", "content": original},
        {"role": "assistant", "content": "command"},
        {"role": "user", "content": token},
    ]
    assert agent._images.project(messages)[2]["content"][1]["type"] == "image_url"
    with pytest.raises(ValueError, match="sole"):
        await agent._execute_commands(
            [SimpleNamespace(keystrokes="gym_view_image plot.png"), SimpleNamespace(keystrokes="echo next")], None
        )
    with pytest.raises(ValueError, match="observation turn"):
        agent._images.project([{"role": "user", "content": token}])
    with pytest.raises(FileExistsError):
        ImageRegistry(tmp_path / "logs/image-observations")
    with pytest.raises(ValueError, match="reuse"):
        await agent.run(original, Environment(), None)


@pytest.mark.parametrize("guard", [False, "true", None])
def test_variant_guard_must_be_authoritative(tmp_path, guard):
    from responses_api_agents.harbor_agent.custom_agents.terminus_2_images_nemo_gym import Terminus2ImagesNemoGym

    with pytest.raises(ValueError, match="authoritative"):
        Terminus2ImagesNemoGym(
            logs_dir=tmp_path,
            model_name="fixture",
            api_base="http://localhost:1/v1",
            gym_provider_dispatch_guard=guard,
            enable_summarize=False,
            image_root="/workspace/results",
        )


def test_pixel_limit_and_forged_token(tmp_path):
    registry = ImageRegistry(tmp_path / "limited", maximum_pixels=1)
    with pytest.raises(ValueError, match="dimensions"):
        registry.register(png())
    with pytest.raises(ValueError, match="Missing"):
        registry.project([{"role": "user", "content": "[[gym-image:" + "a" * 32 + "]]"}])
