# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from resources_servers.molrgen import managed_oracle as module


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["none", "timeout", "cancelled"])
async def test_owned_cleanup_and_resources(tmp_path, monkeypatch, failure):
    sandbox = SimpleNamespace(
        start=AsyncMock(),
        stop=AsyncMock(),
        serialize=AsyncMock(return_value={"id": "owned"}),
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0)),
        endpoint=AsyncMock(return_value=SimpleNamespace(endpoint="http://127.0.0.1:49153")),
    )
    specs = []
    monkeypatch.setattr(module, "AsyncSandbox", lambda provider, spec: specs.append(spec) or sandbox)
    received = AsyncMock()
    received.__aenter__.return_value = received
    received.raise_for_status = lambda: None
    received.json.return_value = {"status": "ready"}
    monkeypatch.setattr(module, "request", AsyncMock(return_value=received))
    (tmp_path / "native.py").write_text("native")
    digest = hashlib.sha256(f"native.py {hashlib.sha256(b'native').hexdigest()}\n".encode()).hexdigest()
    cfg = module.ManagedOracleConfig(
        source_sha256=digest, source_dir=tmp_path, data_dir=tmp_path, artifact_dir=tmp_path / "artifacts"
    )

    async def use():
        async with module.managed_oracle(cfg) as (endpoint, artifacts):
            assert endpoint.endswith("49153")
            if failure == "timeout":
                raise TimeoutError()
            if failure == "cancelled":
                raise asyncio.CancelledError()

    if failure == "none":
        await use()
    else:
        with pytest.raises(TimeoutError if failure == "timeout" else asyncio.CancelledError):
            await use()
    sandbox.stop.assert_awaited_once()
    assert specs[0].resources.cpu == 2 and specs[0].resources.memory_mib == 3072
    assert specs[0].ports == (37000,) and specs[0].ttl_s == 900
    assert specs[0].metadata["workload"] == "molrgen-oracle"
    proof = json.loads(next((tmp_path / "artifacts").glob("*/lifecycle.json")).read_text())
    assert proof["cleanup_confirmed"] and proof["sandbox"] == {"id": "owned"}


@pytest.mark.asyncio
async def test_cancel_during_creation_waits_and_cleans(tmp_path, monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()

    async def start():
        started.set()
        await release.wait()

    sandbox = SimpleNamespace(start=start, stop=AsyncMock())
    monkeypatch.setattr(module, "AsyncSandbox", lambda *args: sandbox)
    (tmp_path / "native.py").write_text("native")
    digest = hashlib.sha256(f"native.py {hashlib.sha256(b'native').hexdigest()}\n".encode()).hexdigest()
    cfg = module.ManagedOracleConfig(
        source_sha256=digest, source_dir=tmp_path, data_dir=tmp_path, artifact_dir=tmp_path / "artifacts"
    )

    async def use():
        async with module.managed_oracle(cfg):
            pass

    task = asyncio.create_task(use())
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    sandbox.stop.assert_awaited_once()
