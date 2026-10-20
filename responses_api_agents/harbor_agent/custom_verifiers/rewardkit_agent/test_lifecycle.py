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
