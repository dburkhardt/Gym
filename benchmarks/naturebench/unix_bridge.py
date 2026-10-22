# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verifier-only loopback relay through Harbor's existing verifier-log mount."""

import asyncio
import json
import shlex
import shutil
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from textwrap import dedent
from typing import AsyncIterator

from aiohttp import web
from harbor.environments.base import BaseEnvironment

from nemo_gym.sandbox import http_unix_relay
from nemo_gym.sandbox.model_bridge import _settle


@asynccontextmanager
async def verifier_unix_bridge(
    runner: web.AppRunner, *, environment: BaseEnvironment, directory: Path
) -> AsyncIterator[str]:
    """Expose the authenticated handler only to the separate verifier's loopback."""
    if not environment.capabilities.mounted:
        raise ValueError("NatureBench Unix bridge requires Harbor's verifier bind mount")
    relay = directory / "relay.py"
    shutil.copyfile(Path(http_unix_relay.__file__), relay)
    container_dir = "/logs/verifier/gym-judge"
    command_path = container_dir + "/relay.py"
    pid = None
    start_task = None
    socket = directory / "model.sock"
    try:
        # Linux limits AF_UNIX names to108 bytes. The socket itself stays inside
        # the existing mount; only the host bind spelling uses a short alias.
        with tempfile.TemporaryDirectory(prefix="ngjudge-", dir="/tmp") as short:
            alias = Path(short) / "v"
            alias.symlink_to(directory.resolve(), target_is_directory=True)
            await web.UnixSite(runner, str(alias / "model.sock")).start()
            socket.chmod(0o600)
            command = (
                f"nohup python {command_path} --socket {container_dir}/model.sock "
                f"--port-file {container_dir}/relay-port.json --route /v1/responses "
                f"--max-body-bytes {32 * 1024 * 1024} "
                f">{container_dir}/relay.log 2>&1 </dev/null & echo $!"
            )
            start_task = asyncio.create_task(environment.exec(command=command, timeout_sec=10))
            started = await _settle(start_task)
            if started.return_code != 0:
                raise RuntimeError("NatureBench verifier relay failed to start")
            pid = int(started.stdout.strip())
            if pid <= 1:
                raise RuntimeError("Invalid verifier relay process identity")
            ready = directory / "relay-port.json"
            async with asyncio.timeout(10):
                while not ready.exists():
                    await asyncio.sleep(0.05)
            value = json.loads(ready.read_text())
            if value.get("pid") != pid or type(value.get("port")) is not int or not 0 < value["port"] < 65536:
                raise ValueError("Verifier relay readiness identity mismatch")
            yield f"http://127.0.0.1:{value['port']}/v1"
    finally:
        try:
            original = sys.exc_info()[1]
            # Startup may finish after cancellation before stdout can be assigned.
            # Reconcile the exact relay command inside this owned verifier container,
            # including an exec timeout whose child-start outcome is uncertain.
            if start_task is not None:
                cleanup = dedent(f"""
    import pathlib,os,signal,time,json
    expected={command_path!r}.encode()
    def owned():
        found=[]
        for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
            try:
                if expected in p.read_bytes().split(b'\\0'):
                    found.append(int(p.parent.name))
            except (FileNotFoundError,ProcessLookupError):
                pass
        return found
    before=owned()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in owned():
            try: os.kill(pid,sig)
            except ProcessLookupError: pass
        deadline=time.monotonic()+2
        while owned() and time.monotonic()<deadline: time.sleep(.02)
    after=owned()
    print(json.dumps({{'before':before,'after':after}}))
    assert not after, 'Owned relay remains alive'
    """)
                task = asyncio.create_task(
                    environment.exec(command="python -c " + shlex.quote(cleanup), timeout_sec=10)
                )
                try:
                    result = await _settle(task)
                    (directory / "relay-cleanup.json").write_text(
                        json.dumps({"return_code": result.return_code, "stdout": result.stdout})
                    )
                    if result.return_code != 0:
                        raise RuntimeError("Owned verifier relay cleanup failed")
                except BaseException as error:
                    (directory / "relay-cleanup-error.json").write_text(
                        json.dumps({"error_type": type(error).__name__})
                    )
                    if original is None:
                        raise
        finally:
            socket.unlink(missing_ok=True)
