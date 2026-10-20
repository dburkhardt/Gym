# SPDX-License-Identifier: Apache-2.0
"""Durable ownership and cancellation-safe cleanup for managed sandboxes."""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from nemo_gym.sandbox import AsyncSandbox


def _publish(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


async def _settle(task: asyncio.Task):
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError()
    return result


@asynccontextmanager
async def _owned_sandbox(sandbox: AsyncSandbox, receipt: Path):
    descriptor = None
    start = asyncio.create_task(sandbox.start())
    try:
        await asyncio.shield(start)
        _publish(receipt, {"handle": await sandbox.serialize(), "cleanup_confirmed": False})
        yield sandbox
    finally:
        original = sys.exc_info()[1]
        try:
            try:
                await _settle(start)
            finally:
                if start.done() and not start.cancelled() and start.exception() is None:
                    descriptor = await sandbox.serialize()
                    _publish(receipt, {"handle": descriptor, "cleanup_confirmed": False})
                await _settle(asyncio.create_task(sandbox.stop()))
            _publish(receipt, {"handle": descriptor, "cleanup_confirmed": True})
        except BaseException as error:
            _publish(receipt.with_suffix(".error.json"), {"type": type(error).__name__, "message": str(error)})
            if original is None:
                raise
