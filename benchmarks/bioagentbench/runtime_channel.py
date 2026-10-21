# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-attempt creator Runtime IPC; the child imports only the Python stdlib.

This is an operation boundary for one parent-owned sandbox, not a scheduler or
recovery protocol. A lost response is unknown and cannot be replayed.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import struct
import time
from pathlib import Path


LIMIT = 64 * 1024 * 1024
METHODS = frozenset({"start", "teardown", "run", "run_background", "read", "write"})


async def read_frame(reader):
    size = struct.unpack("!I", await reader.readexactly(4))[0]
    if size > LIMIT:
        raise ValueError("Runtime IPC frame exceeds admitted limit")
    return json.loads(await reader.readexactly(size))


async def write_frame(writer, value):
    raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(raw) > LIMIT:
        raise ValueError("Runtime IPC frame exceeds admitted limit")
    writer.write(struct.pack("!I", len(raw)) + raw)
    await writer.drain()


class RuntimeClient:
    """Serialized, no-retry child connection; never owns a Docker handle."""

    def __init__(self, socket: Path, token: str, *, deadline: float):
        self.socket, self.token, self.deadline = socket, token, deadline
        self.ordinal = 0
        self.failed = False
        self.lock = asyncio.Lock()

    async def call(self, method, **arguments):
        async with self.lock:
            if self.failed:
                raise RuntimeError("Runtime IPC prior outcome unknown; replay forbidden")
            if method not in METHODS:
                raise ValueError("Unknown Runtime operation")
            if self.deadline <= time.time():
                raise TimeoutError("Runtime absolute deadline expired")
            self.failed = True
            writer = None
            ordinal = self.ordinal
            self.ordinal += 1
            try:
                async with asyncio.timeout(max(0, self.deadline - time.time())):
                    reader, writer = await asyncio.open_unix_connection(str(self.socket))
                    await write_frame(
                        writer, {"token": self.token, "ordinal": ordinal, "method": method, "arguments": arguments}
                    )
                    response = await read_frame(reader)
                    if response.get("ordinal") != ordinal or response.get("ok") is not True:
                        raise RuntimeError("Runtime IPC refused or failed; no automatic retry")
                    self.failed = False
                    return response["result"]
            finally:
                if writer is not None:
                    writer.close()
                    await writer.wait_closed()


class RuntimeServer:
    """Parent endpoint admitting only the six methods on an already fixed backend."""

    def __init__(self, backend, *, socket: Path, token: str, deadline: float, receipts: Path):
        self.backend, self.socket, self.token, self.deadline = backend, socket, token, deadline
        self.receipts = receipts
        self.ordinal = 0
        self.failed = False
        self.started = False
        self.ended = False
        self.lock = asyncio.Lock()
        self.tasks = set()
        self.server = None
        self.socket_identity = None

    async def start(self):
        self.receipts.mkdir(parents=True, exist_ok=False)
        if self.socket.exists() or self.socket.is_symlink():
            raise ValueError("Runtime IPC socket already exists")
        self.server = await asyncio.start_unix_server(self._connection, path=str(self.socket))
        os.chmod(self.socket, 0o600)
        stat = self.socket.lstat()
        self.socket_identity = (stat.st_dev, stat.st_ino)
        return self

    async def _operation(self, method, args):
        if method == "start":
            if self.started or self.ended or args:
                raise ValueError("Runtime can start only once")
            self.started = True
            await self.backend.start()
            return {"id": self.backend.info.id}
        if not self.started or self.ended:
            raise ValueError("Runtime is not active")
        if method == "teardown":
            if args:
                raise ValueError("Unexpected teardown arguments")
            await self.backend.teardown()
            self.ended = True
            return None
        if method in ("run", "run_background"):
            allowed = {"argv", "env"} | ({"log"} if method == "run_background" else set())
            if set(args) != allowed or not isinstance(args["argv"], list) or not args["argv"]:
                raise ValueError("Malformed Runtime command")
            if not all(isinstance(s, str) for s in args["argv"]) or not isinstance(args["env"], dict):
                raise ValueError("Malformed Runtime command")
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in args["env"].items()):
                raise ValueError("Malformed Runtime environment")
            if method == "run_background" and not isinstance(args["log"], str):
                raise ValueError("Malformed Runtime log path")
            result = await getattr(self.backend, method)(**args)
            return (
                None
                if method == "run_background"
                else {k: getattr(result, k) for k in ("exit_code", "stdout", "stderr")}
            )
        if set(args) != ({"path"} if method == "read" else {"path", "data"}) or not isinstance(args["path"], str):
            raise ValueError("Malformed Runtime file operation")
        if method == "read":
            return base64.b64encode(await self.backend.read(args["path"])).decode()
        await self.backend.write(args["path"], base64.b64decode(args["data"], validate=True))
        return None

    async def _connection(self, reader, writer):
        from nemo_gym.sandbox.guarded_chat_transport import publish

        task = asyncio.current_task()
        self.tasks.add(task)
        operation = disconnected = None
        identity = stem = None
        try:
            async with asyncio.timeout(max(0, self.deadline - time.time())):
                request = await read_frame(reader)
                if (
                    not isinstance(request, dict)
                    or not isinstance(request.get("token"), str)
                    or not hmac.compare_digest(request["token"], self.token)
                ):
                    raise ValueError("Runtime IPC authorization failed")
                async with self.lock:
                    ordinal = request.get("ordinal")
                    method, args = request.get("method"), request.get("arguments")
                    if self.failed or type(ordinal) is not int or ordinal != self.ordinal:
                        raise ValueError("Runtime IPC replay, gap or prior unknown outcome")
                    if method not in METHODS or not isinstance(args, dict):
                        raise ValueError("Unknown Runtime operation")
                    self.ordinal += 1
                    self.failed = True
                    identity = {
                        "ordinal": ordinal,
                        "method": method,
                        "arguments_sha256": hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest(),
                    }
                    stem = self.receipts / f"{ordinal:06d}"
                    publish(stem.with_suffix(".intent.json"), identity)
                    operation = asyncio.create_task(self._operation(method, args))
                    disconnected = asyncio.create_task(reader.read(1))
                    done, _ = await asyncio.wait({operation, disconnected}, return_when=asyncio.FIRST_COMPLETED)
                    if disconnected in done:
                        raise RuntimeError("Runtime client disconnected during operation; outcome unknown")
                    result = operation.result()
                    publish(stem.with_suffix(".complete.json"), {**identity, "completed": True})
                    await write_frame(writer, {"ordinal": ordinal, "ok": True, "result": result})
                    self.failed = False
        except BaseException as error:
            if identity is not None and stem is not None:
                # Exception text/arguments may contain private task data or tokens.
                publish(
                    stem.with_suffix(".failure.json"),
                    {**identity, "error_type": type(error).__name__, "outcome": "unresolved_no_replay"},
                )
            # No response retry. Parent teardown settles any partially completed work.
            self.failed = True
        finally:
            for pending in (operation, disconnected):
                if pending is not None:
                    pending.cancel()
            await asyncio.gather(*(p for p in (operation, disconnected) if p is not None), return_exceptions=True)
            writer.close()
            await writer.wait_closed()
            self.tasks.discard(task)

    async def close(self):
        from nemo_gym.sandbox.model_bridge import _settle

        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        for task in tuple(self.tasks):
            task.cancel()

        async def cleanup():
            try:
                await asyncio.gather(*tuple(self.tasks), return_exceptions=True)
            finally:
                try:
                    await self.backend.teardown()
                finally:
                    if self.socket_identity is not None:
                        try:
                            stat = self.socket.lstat()
                        except FileNotFoundError:
                            pass
                        else:
                            if (stat.st_dev, stat.st_ino) != self.socket_identity:
                                raise RuntimeError("Owned Runtime socket was replaced; refusing removal")
                            self.socket.unlink()

        await _settle(asyncio.create_task(cleanup()))
