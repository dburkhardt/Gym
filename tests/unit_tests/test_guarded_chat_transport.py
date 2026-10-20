import json

import httpx
import pytest

from nemo_gym.sandbox.guarded_chat_transport import GuardedChatTransport


@pytest.mark.asyncio
async def test_completed_length_response_preserved_and_calls_distinct(tmp_path):
    seen = []

    async def provider(request):
        seen.append(request.headers["x-nemo-gym-logical-call-id"])
        assert (tmp_path / "calls" / f"{len(seen) - 1:06d}.request.json").exists()
        return httpx.Response(200, json={"model": "model", "choices": [{"finish_reason": "length"}]})

    transport = GuardedChatTransport(
        socket_path="unused",
        token="local",
        lineage="attempt",
        digest="config",
        namespace="flyaoc",
        model="model",
        output=tmp_path / "calls",
        transport=httpx.MockTransport(provider),
    )
    async with httpx.AsyncClient(transport=transport) as client:
        for _ in range(2):
            response = await client.post("http://local/v1/chat/completions", json={"model": "model"})
            assert response.json()["choices"][0]["finish_reason"] == "length"
    assert seen == ["flyaoc:attempt:0", "flyaoc:attempt:1"]
    assert json.loads((tmp_path / "calls/000001.response.json").read_text())["status"] == 200
    with pytest.raises(FileExistsError):
        GuardedChatTransport(
            socket_path="unused",
            token="local",
            lineage="attempt",
            digest="config",
            namespace="flyaoc",
            model="model",
            output=tmp_path / "calls",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "status"])
async def test_ambiguous_or_error_response_cannot_be_retried(tmp_path, failure):
    calls = 0

    async def provider(request):
        nonlocal calls
        calls += 1
        if failure == "timeout":
            raise httpx.ReadTimeout("accepted then connection lost")
        return httpx.Response(500, json={"error": "failure"})

    transport = GuardedChatTransport(
        socket_path="unused",
        token="local",
        lineage="attempt",
        digest="config",
        namespace="flyaoc",
        model="model",
        output=tmp_path / "calls",
        transport=httpx.MockTransport(provider),
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises((RuntimeError, httpx.ReadTimeout)):
            await client.post("http://local/v1/chat/completions", json={"model": "model"})
        with pytest.raises(RuntimeError, match="no automatic resend"):
            await client.post("http://local/v1/chat/completions", json={"model": "model"})
    assert calls == 1
