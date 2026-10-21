# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest


pytest.importorskip("verifiers.v1", reason="Pinned optional creator runtime is not installed")

from benchmarks.bioagentbench import runtime


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    (source / "reference").mkdir()
    (source / "results").mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    events = []

    class Sandbox:
        def __init__(self, provider, spec):
            self.provider, self.spec = provider, spec

        async def serialize(self):
            return {"sandbox_id": "owned-immutable-container"}

        async def exec(self, command, timeout_s):
            events.append((command, timeout_s))
            return SimpleNamespace(error_type=None, return_code=0, stdout="", stderr="")

    @asynccontextmanager
    async def owner(sandbox, receipt):
        events.append("created")
        try:
            yield sandbox
        finally:
            events.append("removed")

    monkeypatch.setattr(runtime, "AsyncSandbox", Sandbox)
    monkeypatch.setattr(runtime, "_owned_sandbox", owner)
    cls = runtime.runtime_class(
        workspace=workspace, source_dir=source, receipt=tmp_path / "receipt.json", image="sha256:" + "a" * 64
    )
    from verifiers.v1.runtimes.docker import DockerConfig

    return cls(DockerConfig(cpu=2, memory=8)), events


def test_native_contract_limits_mounts(prepared):
    instance, _ = prepared
    assert instance.sandbox.provider["docker"]["create"]["network"] == "bridge"
    mounts = instance.sandbox.spec.provider_options["volumes"]
    assert len(mounts) == 3
    assert all("results" not in mount for mount in mounts)
    assert all(not mount.endswith(":ro") for mount in mounts[1:])  # owned input copies support creator index writes
    assert instance.sandbox.spec.resources.memory_mib == 8192
    assert instance.is_local is False
    assert instance.info.image.startswith("sha256:")


@pytest.mark.asyncio
async def test_native_start_stop_is_owned_and_idempotent(prepared):
    instance, events = prepared
    await instance.start()
    assert instance.info.id == "owned-immutable-container"
    await instance.stop()
    await instance.stop()
    assert events == ["created", "removed"]


@pytest.mark.asyncio
async def test_cancellation_reaches_native_finally(prepared):
    instance, events = prepared

    async def task():
        await instance.start()
        try:
            await asyncio.Event().wait()
        finally:
            await instance.stop()

    pending = asyncio.create_task(task())
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert events == ["created", "removed"]


@pytest.mark.asyncio
async def test_timeout_is_not_scientific_zero(prepared):
    instance, _ = prepared

    async def timeout(*args, **kwargs):
        return SimpleNamespace(error_type="timeout")

    instance.sandbox.exec = timeout
    with pytest.raises(RuntimeError, match="timeout"):
        await instance.run(["scientific-solver"], {})


def test_reject_mutable_image_and_public_symlink(tmp_path):
    kwargs = dict(workspace=tmp_path, source_dir=tmp_path, receipt=tmp_path / "receipt")
    with pytest.raises(ValueError, match="immutable"):
        runtime.runtime_class(**kwargs, image="python:latest")
    (tmp_path / "data").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlinks"):
        runtime.runtime_class(**kwargs, image="sha256:" + "a" * 64)


@pytest.mark.asyncio
async def test_download_refuses_solver_symlink_to_host(prepared, tmp_path):
    instance, _ = prepared
    secret = tmp_path / "host-file"
    secret.write_text("must not be read")

    async def download(remote, local):
        local.symlink_to(secret)

    instance.sandbox.download = download
    with pytest.raises(ValueError, match="regular file"):
        await instance.read("results/evil-link")
