# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cancellation must settle creation before destroying its owned sandbox."""

import asyncio
import json

import pytest

from responses_api_agents.harbor_agent.custom_verifiers.rewardkit_agent.verifier import _owned_sandbox


@pytest.mark.asyncio
async def test_cancelled_creation_settles_before_cleanup(tmp_path):
    created = asyncio.Event()
    release = asyncio.Event()
    calls = []

    class Sandbox:
        async def start(self):
            created.set()
            await release.wait()
            calls.append("created")

        async def serialize(self):
            return {"sandbox_id": "owned-id"}

        async def stop(self):
            assert calls == ["created"]
            calls.append("removed")

    async def run():
        async with _owned_sandbox(Sandbox(), tmp_path / "receipt.json"):
            pytest.fail("Cancelled creation must not enter the body")

    task = asyncio.create_task(run())
    await created.wait()
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == ["created", "removed"]
    assert json.loads((tmp_path / "receipt.json").read_text())["cleanup_confirmed"]


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_mask_original(tmp_path):
    class Sandbox:
        async def start(self):
            pass

        async def serialize(self):
            return {"sandbox_id": "owned-id"}

        async def stop(self):
            raise RuntimeError("cleanup failed")

    with pytest.raises(ValueError, match="measurement failed"):
        async with _owned_sandbox(Sandbox(), tmp_path / "receipt.json"):
            raise ValueError("measurement failed")
    assert json.loads((tmp_path / "receipt.error.json").read_text())["message"] == "cleanup failed"


def test_judge_request_passes_native_gym_validation():
    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from responses_api_agents.harbor_agent.custom_verifiers.rewardkit_agent.tool_loop import build_request

    request = build_request(
        [{"role": "user", "content": "Inspect the submitted workspace"}],
        {
            "type": "object",
            "properties": {"pass": {"type": "boolean"}},
            "required": ["pass"],
            "additionalProperties": False,
        },
    )
    parsed = NeMoGymResponseCreateParamsNonStreaming.model_validate(request)
    assert parsed.tools[0]["strict"] is False
    assert parsed.max_output_tokens == 16384


@pytest.mark.parametrize("runtime", [None, "runc"])
@pytest.mark.asyncio
async def test_runtime_applies_to_both_owned_verifier_containers(tmp_path, monkeypatch, runtime):
    from types import SimpleNamespace

    from responses_api_agents.harbor_agent.custom_verifiers.rewardkit_agent import verifier as module

    image = "sha256:" + "a" * 64
    tests = tmp_path / "tests"
    tests.mkdir()
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "trajectory.json").write_text("{}")
    config = SimpleNamespace(
        environment=SimpleNamespace(docker_image=image, cpus=4, memory_mb=8192),
        verifier=SimpleNamespace(timeout_sec=900),
        model_dump=lambda **kwargs: {},
    )

    def initialize(self, **kwargs):
        self.task = SimpleNamespace(config=config, paths=SimpleNamespace(tests_dir=tests))
        self.trial_paths = SimpleNamespace(verifier_dir=tmp_path / "verifier", agent_dir=agent)

        async def download(source, destination):
            destination.mkdir()

        self.environment = SimpleNamespace(download_dir=download)

    monkeypatch.setattr(module.BaseVerifier, "__init__", initialize)
    observed = []

    class Sandbox:
        def __init__(self, options, spec):
            observed.append(options)

        async def start(self):
            pass

        async def stop(self):
            pass

        async def serialize(self):
            return {"sandbox_id": "owned-test"}

        async def exec(self, command, timeout_s):
            result = tmp_path / "verifier/native-controller/controller/result"
            result.mkdir()
            (result / "classification.json").write_text(
                json.dumps({"mask_sample": False, "native_reward": {"reward": 1.0}})
            )
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    monkeypatch.setattr(module, "AsyncSandbox", Sandbox)
    instance = module.RewardKitAgentVerifier(
        image_id=image,
        gym_model_api_base="http://localhost/ng-rollout/test/v1",
        gym_model_name="nvidia/nvidia/nemotron-3.5-super-vl",
        docker_runtime=runtime,
    )
    await instance.verify()
    assert len(observed) == 2
    assert [item["docker"]["create"]["network"] for item in observed] == ["none", "host"]
    for item in observed:
        assert item["docker"]["create"]["extra_run_args"] == ([] if runtime is None else ["--runtime", "runc"])


@pytest.mark.parametrize("runtime", ["", "--privileged", "runc --gpus all", 42])
def test_reject_invalid_runtime_before_environment_setup(runtime):
    from responses_api_agents.harbor_agent.custom_verifiers.rewardkit_agent.verifier import RewardKitAgentVerifier

    with pytest.raises(ValueError, match="docker_runtime"):
        RewardKitAgentVerifier(
            image_id="unused", gym_model_api_base="unused", gym_model_name="unused", docker_runtime=runtime
        )
