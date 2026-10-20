# SPDX-License-Identifier: Apache-2.0
import sys
import types

import pytest

from benchmarks.corral.guarded_agent import MODEL, guarded_agent


@pytest.mark.asyncio
async def test_durable_call_binding_and_no_retry(tmp_path, monkeypatch):
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
    with guarded_agent(endpoint, "fresh-lineage", tmp_path / "calls"):
        args = {"model": "openai/" + MODEL, "api_endpoint": endpoint, "max_tokens": 16384, "temperature": 1.0}
        with pytest.raises(TimeoutError):
            await base.llm_call(**args)
        with pytest.raises(RuntimeError, match="unresolved"):
            await base.llm_call(**args)
    assert len(calls) == 1
    assert calls[0]["extra_headers"]["x-nemo-gym-logical-call-id"] == "corral:fresh-lineage:0"
    assert (tmp_path / "calls/000000.intent.json").exists()
    assert (tmp_path / "calls/000000.failure.json").exists()
    assert base.llm_call is original
