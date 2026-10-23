# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import sys
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from responses_api_agents.metasyn_native.app import Config
from responses_api_agents.metasyn_native.isolation import generator_spec
from responses_api_agents.metasyn_native.public_generator import generate


def test_live_configuration_requires_immutable_isolation_image():
    fields = dict(
        host="127.0.0.1",
        port=1,
        name="metasyn",
        entrypoint="app.py",
        native_python="python",
        runner_script=Path("runner"),
        materialized_root=Path("data"),
        output_root=Path("out"),
        reference_file=Path("refs"),
    )
    assert Config(**fields).inference_enabled is False
    with pytest.raises(ValidationError, match="public_generator_image"):
        Config(**fields, inference_enabled=True)
    with pytest.raises(ValidationError):
        Config(**fields, inference_enabled=True, public_generator_image="mutable:latest")
    assert Config(**fields, inference_enabled=True, public_generator_image="sha256:" + "a" * 64).inference_enabled


def test_generator_preserves_payload_and_rejects_extra_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    public = tmp_path / "public"
    public.mkdir()
    (public / "rag.py").write_text(
        'from openai import OpenAI\ndef generate_report(prompt, model, api_key):\n c=OpenAI(api_key=api_key).chat.completions.create(model=model,messages=[{"role":"user","content":prompt}])\n return c.choices[0].message.content, c.usage.model_dump()\n'
    )
    (public / "prompt.txt").write_text("public prompt")
    expected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in public.iterdir()}
    output = tmp_path / "output"
    output.mkdir()
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 0,
                "model": "fixture",
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "saved report"}}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    generate(public, output, expected, transport=httpx.MockTransport(respond))
    assert calls == [
        {"model": "nvidia/nvidia/nemotron-3.5-super-vl", "messages": [{"role": "user", "content": "public prompt"}]}
    ]
    result = json.loads((output / "generation.json").read_text())
    assert result["report"] == "saved report" and result["usage"]["total_tokens"] == 5
    with pytest.raises(FileExistsError):
        generate(public, output, expected)
    (public / "references.jsonl").write_text("gold")
    with pytest.raises(ValueError, match="binding"):
        generate(public, output, expected)
    assert len(calls) == 1


@pytest.mark.parametrize("remaining", [0, -1, float("nan"), float("inf")])
def test_expired_or_invalid_budget_refuses_sandbox_spec(tmp_path, remaining):
    with pytest.raises(ValueError, match="remaining phase"):
        generator_spec(
            "sha256:" + "a" * 64,
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path / "socket",
            remaining=remaining,
            lineage="test",
        )


@pytest.mark.asyncio
async def test_prompt_changed_after_parent_identity_refuses_before_bridge(tmp_path, monkeypatch):
    from responses_api_agents.metasyn_native import isolation

    public = tmp_path / "generator-public"

    def export(*args):
        public.mkdir()
        return {"prompt.txt": "changed"}

    monkeypatch.setattr(isolation, "export_public", export)
    with pytest.raises(ValueError, match="parent scientific identity"):
        await isolation.run_generator(
            image="sha256:" + "a" * 64,
            creator=tmp_path,
            prompt=tmp_path / "prompt",
            out=tmp_path,
            endpoint="http://unused",
            lineage="test",
            digest="digest",
            deadline=0,
            expected_prompt_sha256="original",
        )
    assert not (tmp_path / "generator-control").exists()


@pytest.mark.asyncio
async def test_deadline_expiry_during_startup_never_executes_and_cleans(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from nemo_gym.sandbox import model_bridge
    from responses_api_agents.metasyn_native import isolation

    events = []
    clock = [0.0]
    socket_dir = tmp_path / "socket"
    socket_dir.mkdir()
    out = tmp_path / "attempt"
    out.mkdir()

    def export(*args):
        public = out / "generator-public"
        public.mkdir()
        (public / "rag.py").write_text("public")
        (public / "prompt.txt").write_text("prompt")
        return {"prompt.txt": "expected"}

    @asynccontextmanager
    async def workspace(provider):
        yield SimpleNamespace(raw={"workspace": socket_dir})

    @asynccontextmanager
    async def bridge(socket, **kwargs):
        socket.touch()
        yield

    class Box:
        async def serialize(self):
            return {"id": "owned-fixture"}

        async def exec(self, *args, **kwargs):
            raise AssertionError("Expired container must not execute")

    @asynccontextmanager
    async def owned(box, receipt):
        events.append("start")
        clock[0] = 11
        try:
            yield
        finally:
            events.append("cleanup")

    monkeypatch.setattr(isolation, "export_public", export)
    monkeypatch.setattr(isolation, "scoring_workspace", workspace)
    monkeypatch.setattr(model_bridge, "bridge", bridge)
    monkeypatch.setattr(isolation, "DockerProvider", lambda **kwargs: object())
    monkeypatch.setattr(isolation, "AsyncSandbox", lambda *args: Box())
    monkeypatch.setattr(isolation, "_owned_sandbox", owned)
    monkeypatch.setattr(isolation.time, "time", lambda: clock[0])
    with pytest.raises(TimeoutError, match="during container startup"):
        await isolation.run_generator(
            image="sha256:" + "a" * 64,
            creator=tmp_path,
            prompt=tmp_path / "prompt",
            out=out,
            endpoint="http://unused",
            lineage="test",
            digest="digest",
            deadline=10,
            expected_prompt_sha256="expected",
        )
    assert events == ["start", "cleanup"]
    assert not (out / "generator-execution.json").exists()
