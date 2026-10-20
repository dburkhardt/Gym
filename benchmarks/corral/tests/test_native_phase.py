# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace

import pytest

from benchmarks.corral.native_phase import _close_store


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_persistence_close():
    started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def close():
        started.set()
        await release.wait()
        finished.set()

    cleanup = asyncio.create_task(_close_store(SimpleNamespace(aclose=close)))
    await started.wait()
    cleanup.cancel()
    await asyncio.sleep(0)
    cleanup.cancel()
    await asyncio.sleep(0)
    assert not finished.is_set()
    assert not cleanup.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await cleanup
    assert finished.is_set()
