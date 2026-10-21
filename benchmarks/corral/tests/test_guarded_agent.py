# SPDX-License-Identifier: Apache-2.0
import sys
import types

import pytest

from benchmarks.corral.guarded_agent import MODEL, guarded_agent


@pytest.mark.asyncio
@pytest.mark.parametrize("digest", [None, "1" * 64])
async def test_durable_call_binding_and_no_retry(tmp_path, monkeypatch, digest):
    calls = []

    async def original(**kwargs):
        calls.append(kwargs)
        raise TimeoutError("ambiguous")

    base = types.ModuleType("corral.agents.base_agent")
    base.llm_call = original
    agents = types.ModuleType("corral.agents")
    agents.base_agent = base
    root = types.ModuleType("corral")
    root.agents = agents
    cli = types.ModuleType("corral.cli")
    cli.create_agent = lambda *args, **kwargs: kwargs
    for name, module in [
        ("corral", root),
        ("corral.agents", agents),
        ("corral.agents.base_agent", base),
        ("corral.cli", cli),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    endpoint = "http://model/ng-rollout/native/v1"
    with guarded_agent(endpoint, "fresh-lineage", tmp_path / "calls", dispatch_digest=digest):
        args = {"model": "openai/" + MODEL, "api_endpoint": endpoint, "max_tokens": 16384, "temperature": 1.0}
        with pytest.raises(TimeoutError):
            await base.llm_call(**args)
        with pytest.raises(RuntimeError, match="unresolved"):
            await base.llm_call(**args)
    if digest is not None:
        assert calls[0]["extra_headers"]["x-nemo-gym-call-config-sha256"] == digest
    assert len(calls) == 1
    assert calls[0]["extra_headers"]["x-nemo-gym-logical-call-id"] == "corral:fresh-lineage:0"
    assert (tmp_path / "calls/000000.intent.json").exists()
    assert (tmp_path / "calls/000000.failure.json").exists()
    assert base.llm_call is original


def test_provider_message_id_normalization_preserves_scientific_history():
    from copy import deepcopy

    from benchmarks.corral.guarded_agent import provider_messages

    messages = [
        {"role": "user", "content": "frozen prompt", "id": "do-not-silently-normalize-user"},
        {
            "role": "assistant",
            "id": "provider-response",
            "content": "native action",
            "tool_calls": [{"id": "tool-1", "type": "function", "function": {"name": "tool", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "tool-1", "content": "observation"},
    ]
    before = deepcopy(messages)
    result = provider_messages(messages)
    assert messages == before
    assert result[0] == before[0] and result[2] == before[2]
    assert result[1] == {key: value for key, value in before[1].items() if key != "id"}
    assert result[1]["tool_calls"][0]["id"] == "tool-1"


def test_normalized_assistant_passes_native_chat_schema():
    from pydantic import ValidationError

    from benchmarks.corral.guarded_agent import provider_messages
    from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming

    schema = NeMoGymChatCompletionCreateParamsNonStreaming
    message = {"role": "assistant", "id": "provider-response", "content": "native action"}
    with pytest.raises(ValidationError):
        schema.model_validate({"model": MODEL, "messages": [message]})
    assert (
        schema.model_validate({"model": MODEL, "messages": provider_messages([message])}).messages[0]["content"]
        == "native action"
    )
