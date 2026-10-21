# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import time
from types import SimpleNamespace

import pytest

from benchmarks.bioagentbench.runtime_channel import RuntimeClient, RuntimeServer, read_frame, write_frame


class Backend:
    def __init__(self):
        self.info = SimpleNamespace(id="owned-container")
        self.calls = []
        self.files = {}
        self.wait = None
        self.settled = asyncio.Event()

    async def start(self):
        self.calls.append(("start",))

    async def teardown(self):
        self.calls.append(("teardown",))

    async def run(self, argv, env):
        self.calls.append(("run", argv, env))
        if self.wait is not None:
            try:
                await self.wait.wait()
            finally:
                self.settled.set()
        return SimpleNamespace(exit_code=3, stdout="scientific stdout", stderr="scientific stderr")

    async def run_background(self, argv, env, log):
        self.calls.append(("run_background", argv, env, log))

    async def write(self, path, data):
        self.files[path] = data

    async def read(self, path):
        return self.files[path]


async def setup(tmp_path):
    b = Backend()
    s = await RuntimeServer(
        b, socket=tmp_path / "s", token="scoped-secret", deadline=time.time() + 10, receipts=tmp_path / "receipts"
    ).start()
    return b, s, RuntimeClient(tmp_path / "s", "scoped-secret", deadline=time.time() + 10)


@pytest.mark.asyncio
async def test_six_methods_preserve_values_and_teardown(tmp_path):
    b, s, c = await setup(tmp_path)
    try:
        assert await c.call("start") == {"id": "owned-container"}
        data = b"\x00\xffscientific\n"
        await c.call("write", path="/app/a.csv", data=base64.b64encode(data).decode())
        assert base64.b64decode(await c.call("read", path="/app/a.csv")) == data
        assert await c.call("run", argv=["python", "-c", "print('a b')"], env={"X": "a b"}) == {
            "exit_code": 3,
            "stdout": "scientific stdout",
            "stderr": "scientific stderr",
        }
        await c.call("run_background", argv=["server", "a b"], env={}, log="a b.log")
        await c.call("teardown")
        assert b.calls[1] == ("run", ["python", "-c", "print('a b')"], {"X": "a b"})
        assert len(list((tmp_path / "receipts").glob("*.complete.json"))) == 6
    finally:
        await s.close()
    assert not (tmp_path / "s").exists()


@pytest.mark.asyncio
async def test_replay_refused_without_second_effect(tmp_path):
    b, s, c = await setup(tmp_path)
    try:
        await c.call("start")
        r, w = await asyncio.open_unix_connection(str(tmp_path / "s"))
        await write_frame(w, {"token": "scoped-secret", "ordinal": 0, "method": "start", "arguments": {}})
        with pytest.raises(asyncio.IncompleteReadError):
            await read_frame(r)
        w.close()
        await w.wait_closed()
        assert b.calls == [("start",)]
        with pytest.raises(asyncio.IncompleteReadError):
            await c.call("run", argv=["echo"], env={})
        assert s.failed
    finally:
        await s.close()


@pytest.mark.asyncio
async def test_disconnect_poison_cancels_and_settles_before_teardown(tmp_path):
    b, s, c = await setup(tmp_path)
    try:
        await c.call("start")
        b.wait = asyncio.Event()
        task = asyncio.create_task(c.call("run", argv=["science"], env={}))
        while len(b.calls) < 2:
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(b.settled.wait(), 1)
        assert s.failed and c.failed
        with pytest.raises(RuntimeError, match="replay forbidden"):
            await c.call("run", argv=["science"], env={})
    finally:
        await s.close()
    assert b.calls[-1] == ("teardown",)
    assert not (tmp_path / "s").exists()


@pytest.mark.asyncio
async def test_expired_deadline_does_not_dispatch(tmp_path):
    b, s, c = await setup(tmp_path)
    c.deadline = time.time() - 1
    try:
        with pytest.raises(TimeoutError):
            await c.call("start")
        assert b.calls == []
    finally:
        await s.close()


@pytest.mark.asyncio
async def test_unknown_method_refused(tmp_path):
    b, s, c = await setup(tmp_path)
    try:
        with pytest.raises(ValueError, match="Unknown"):
            await c.call("docker", command="foreign")
        assert b.calls == []
    finally:
        await s.close()


@pytest.mark.asyncio
async def test_completed_operation_lost_reply_is_not_replayed(tmp_path, monkeypatch):
    from benchmarks.bioagentbench import runtime_channel

    b, s, c = await setup(tmp_path)
    original = runtime_channel.write_frame

    async def lose_reply(writer, value):
        if value.get("ok") is True:
            raise ConnectionResetError("delivery lost after operation")
        await original(writer, value)

    monkeypatch.setattr(runtime_channel, "write_frame", lose_reply)
    try:
        with pytest.raises(asyncio.IncompleteReadError):
            await c.call("start")
        assert b.calls == [("start",)]
        assert (tmp_path / "receipts/000000.complete.json").exists()
        assert c.failed and s.failed
        with pytest.raises(RuntimeError, match="replay forbidden"):
            await c.call("start")
    finally:
        await s.close()


@pytest.mark.asyncio
async def test_replaced_socket_is_not_removed(tmp_path):
    b, s, c = await setup(tmp_path)
    (tmp_path / "s").unlink()
    (tmp_path / "s").write_text("unrelated replacement")
    with pytest.raises(RuntimeError, match="replaced"):
        await s.close()
    assert (tmp_path / "s").read_text() == "unrelated replacement"
    assert b.calls == [("teardown",)]


@pytest.mark.asyncio
async def test_backend_failure_is_recorded_without_private_text_and_remains_poisoned(tmp_path):
    import json

    b, s, c = await setup(tmp_path)
    try:
        await c.call("start")
        with pytest.raises(asyncio.IncompleteReadError):
            await c.call("read", path="private-secret-path")
        receipt = json.loads((tmp_path / "receipts/000001.failure.json").read_text())
        assert receipt["error_type"] == "KeyError"
        assert receipt["outcome"] == "unresolved_no_replay"
        assert "private-secret-path" not in json.dumps(receipt)
        assert not (tmp_path / "receipts/000001.complete.json").exists()
        with pytest.raises(RuntimeError, match="replay forbidden"):
            await c.call("read", path="private-secret-path")
    finally:
        await s.close()
