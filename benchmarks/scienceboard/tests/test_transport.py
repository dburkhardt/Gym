# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path

import httpx
import pytest

from benchmarks.scienceboard.guarded_model import CreatorModelHTTP, bind_creator_model
from benchmarks.scienceboard.result_mapping import read_creator_result


SOURCE = Path(
    os.environ.get(
        "SCIENCEBOARD_SOURCE_ROOT",
        str(
            Path(__file__).resolve().parents[5]
            / "outputs/broad-eval-campaign/harbor/agent-expansion/sources/223/OS-Copilot-ScienceBoard-21b5eb0"
        ),
    )
)
MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


def test_actual_creator_image_payload_and_parser(tmp_path):
    native = pytest.importorskip("sci.base.model")
    from PIL import Image

    observed = []

    def respond(request):
        observed.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": MODEL,
                "choices": [{"message": {"role": "assistant", "content": "WAIT(1)"}, "finish_reason": "stop"}],
            },
        )

    transport = CreatorModelHTTP(
        socket_path="/unused",
        token="fixture",
        lineage="task-1",
        digest="a" * 64,
        model=MODEL,
        output=tmp_path / "calls",
        inner=httpx.MockTransport(respond),
    )
    model = native.Model(
        model_style="openai",
        base_url="http://gym-internal/chat/completions",
        model_name=MODEL,
        max_tokens=1500,
        top_p=None,
        temperature=0,
    )
    screenshot = native.ImageContent(Image.new("RGB", (2, 3)))
    message = model.message("user", [native.TextContent("Exact task"), screenshot])._asdict()
    with bind_creator_model(native, SOURCE, transport):
        response = model([message], 480)
        parsed = model.access(response, 1)
        assert parsed.content[0].text == "WAIT(1)"
    assert observed == [{"model": MODEL, "messages": [message], "max_tokens": 1500, "temperature": 0}]
    assert list((tmp_path / "calls").glob("*.response.json"))


def test_failure_poison_stops_creator_retry(tmp_path):
    count = []

    def fail(request):
        count.append(request)
        raise httpx.ReadTimeout("fixture timeout")

    t = CreatorModelHTTP(
        socket_path="/unused",
        token="fixture",
        lineage="task",
        digest="b" * 64,
        model=MODEL,
        output=tmp_path / "calls",
        inner=httpx.MockTransport(fail),
    )
    args = dict(
        headers={"Content-Type": "application/json"}, proxies=None, json={"model": MODEL, "messages": []}, timeout=480
    )
    try:
        with pytest.raises(httpx.ReadTimeout):
            t.post("http://gym-internal/chat/completions", **args)
        with pytest.raises(RuntimeError, match="automatic retry"):
            t.post("http://gym-internal/chat/completions", **args)
        assert len(count) == 1
        mapped = read_creator_result(tmp_path / "missing.out", task_id="actual", transport_outcome=t.outcome)
        assert mapped["mask_sample"] and not mapped["scored"]
    finally:
        t.close()


def test_result_missing_is_not_zero_and_native_zero_retained(tmp_path):
    p = tmp_path / "result.out"
    assert read_creator_result(p, task_id="actual")["raw_reward"] is None
    p.write_text("0")
    assert read_creator_result(p, task_id="actual")["raw_reward"] == 0
    assert not read_creator_result(p, task_id="actual")["mask_sample"]
    rejected = read_creator_result(p, task_id="actual", creator_error="verifier raised")
    assert rejected["mask_sample"] and rejected["raw_reward"] == 0


@pytest.mark.asyncio
async def test_body_timeout_releases_response():
    import asyncio

    from benchmarks.scienceboard.guarded_model import BoundedUDS

    released = []

    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"partial"
            await asyncio.sleep(10)

        async def aclose(self):
            released.append(True)

    transport = BoundedUDS(
        "/unused", timeout=0.01, inner=httpx.MockTransport(lambda request: httpx.Response(200, stream=SlowBody()))
    )
    try:
        with pytest.raises(TimeoutError):
            await transport.handle_async_request(httpx.Request("POST", "http://gym-internal/chat/completions"))
        assert released
    finally:
        await transport.aclose()


def test_null_native_action_retained_without_resend(tmp_path):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "model": MODEL,
                "choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "length"}],
            },
        )

    t = CreatorModelHTTP(
        socket_path="/unused",
        token="fixture",
        lineage="task",
        digest="c" * 64,
        model=MODEL,
        output=tmp_path / "calls",
        inner=httpx.MockTransport(respond),
    )
    args = dict(
        headers={"Content-Type": "application/json"}, proxies=None, json={"model": MODEL, "messages": []}, timeout=480
    )
    try:
        with pytest.raises(ValueError, match="no textual action"):
            t.post("http://gym-internal/chat/completions", **args)
        with pytest.raises(RuntimeError, match="automatic retry"):
            t.post("http://gym-internal/chat/completions", **args)
        saved = json.loads(next((tmp_path / "calls").glob("*.response.json")).read_text())
        assert json.loads(saved["body"])["choices"][0]["message"]["content"] is None
        assert len(calls) == 1
        mapped = read_creator_result(tmp_path / "missing.out", task_id="actual", transport_outcome=t.outcome)
        assert mapped["attempted"] and not mapped["mask_sample"] and not mapped["scored"]
        assert mapped["disposition"] == "model_output_budget"
    finally:
        t.close()


def test_model_failure_mapping_is_not_transport_exclusion(tmp_path):
    missing = tmp_path / "result.out"
    budget = read_creator_result(missing, task_id="actual", transport_outcome="model_output_budget")
    assert budget["attempted"] and not budget["mask_sample"]
    assert budget["raw_reward"] is None and not budget["scored"]
    unknown = read_creator_result(missing, task_id="actual", transport_outcome="transport_unresolved")
    assert unknown["attempted"] and unknown["mask_sample"] and not unknown["scored"]
    missing.write_bytes(b"corrupt\xff")
    bad = read_creator_result(missing, task_id="actual")
    assert bad["disposition"] == "verifier_artifact_failure" and bad["result_sha256"]
    assert missing.read_bytes() == b"corrupt\xff"


def test_filtered_provider_reply_is_not_model_invalid_action(tmp_path):
    transport = CreatorModelHTTP(
        socket_path="unused",
        token="fixture",
        lineage="filtered",
        digest="a" * 64,
        model=MODEL,
        output=tmp_path / "calls",
        inner=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"model": MODEL, "choices": [{"message": {"content": None}, "finish_reason": "content_filter"}]},
            )
        ),
    )
    try:
        with pytest.raises(ValueError):
            transport.post(
                "http://gym-internal/chat/completions",
                headers={},
                proxies=None,
                json={"model": MODEL, "messages": []},
                timeout=480,
            )
        assert transport.outcome == "provider_content_filter"
        result = read_creator_result(tmp_path / "missing", task_id="id", transport_outcome=transport.outcome)
        assert result["raw_reward"] is None and result["mask_sample"] is True
        assert result["disposition"] == "provider_content_filter"
    finally:
        transport.close()
