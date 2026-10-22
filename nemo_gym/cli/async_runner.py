# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graceful cancellation at synchronous evaluation CLI boundaries."""

import asyncio
import signal
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from types import FrameType
from typing import TypeVar


_Result = TypeVar("_Result")


@contextmanager
def sigterm_cleanup_scope() -> Iterator[Callable[[Coroutine[object, object, _Result]], _Result]]:
    """Own SIGTERM through asynchronous work and outer synchronous cleanup.

    Cancel the active task once; repeated SIGTERM must not interrupt finalizers.
    The external supervisor remains responsible for bounded SIGKILL fallback.
    SIGINT handling remains owned by asyncio.run.
    """
    terminated = False
    active: asyncio.Task | None = None
    loop: asyncio.AbstractEventLoop | None = None

    def terminate(_signum: int, _frame: FrameType | None) -> None:
        nonlocal terminated
        if not terminated:
            terminated = True
            if active is not None and loop is not None:
                loop.call_soon_threadsafe(active.cancel)

    def run(coroutine: Coroutine[object, object, _Result]) -> _Result:
        async def work() -> _Result:
            nonlocal active, loop
            active = asyncio.current_task()
            loop = asyncio.get_running_loop()
            try:
                if terminated:
                    coroutine.close()
                    raise asyncio.CancelledError
                return await coroutine
            finally:
                active = None
                loop = None

        return asyncio.run(work())

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        yield run
    finally:
        signal.signal(signal.SIGTERM, previous)
        if terminated:
            # A collector must not turn swallowed termination into success.
            raise SystemExit(128 + signal.SIGTERM) from None


def run_with_sigterm_cleanup(coroutine: Coroutine[object, object, _Result]) -> _Result:
    """Run a standalone coroutine within a SIGTERM cleanup scope."""
    with sigterm_cleanup_scope() as run:
        return run(coroutine)
