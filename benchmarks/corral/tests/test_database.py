# SPDX-License-Identifier: Apache-2.0
import hashlib
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from benchmarks.corral import database


@pytest.mark.asyncio
async def test_snapshot_mismatch_precedes_allocation(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.write_bytes(b"fixture")
    monkeypatch.setattr(database, "AsyncSandbox", lambda *a: pytest.fail("allocated"))
    with pytest.raises(ValueError, match="checksum"):
        async with database.database_clone(
            snapshot=snapshot, expected_sha256="0" * 64, socket_dir=tmp_path / "socket", receipt=tmp_path / "receipt"
        ):
            pytest.fail("yielded")


@pytest.mark.asyncio
async def test_restore_failure_settles_owned_cleanup(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.write_bytes(b"fixture")
    calls, cleanup = [], []

    class Fake:
        def __init__(self, provider, spec):
            assert provider["docker"]["create"]["network"] == "none"
            assert spec.resources.memory_mib == 2048

        async def exec(self, command, **kwargs):
            calls.append(command)
            return SimpleNamespace(return_code=0 if len(calls) == 1 else 1, stderr="restore failed")

    @asynccontextmanager
    async def owned(*args):
        try:
            yield
        finally:
            cleanup.append(True)

    monkeypatch.setattr(database, "AsyncSandbox", Fake)
    monkeypatch.setattr(database, "_owned_sandbox", owned)
    with pytest.raises(RuntimeError, match="restore failed"):
        async with database.database_clone(
            snapshot=snapshot,
            expected_sha256=hashlib.sha256(b"fixture").hexdigest(),
            socket_dir=tmp_path / "socket",
            receipt=tmp_path / "receipt",
        ):
            pytest.fail("yielded")
    assert cleanup == [True]
    assert len(calls) == 2
