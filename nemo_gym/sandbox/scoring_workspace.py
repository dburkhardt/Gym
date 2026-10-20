# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned local workspace lifecycle for trusted scoring subprocesses."""

import asyncio
import sys
from contextlib import asynccontextmanager

from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.sandbox.providers.local.provider import LocalProvider


@asynccontextmanager
async def scoring_workspace(provider: LocalProvider):
    handle = await provider.create(SandboxSpec())
    try:
        yield handle
    finally:
        original = sys.exception()
        cleanup = asyncio.create_task(provider.close(handle))
        interrupted = None
        while True:
            try:
                await asyncio.shield(cleanup)
                break
            except asyncio.CancelledError as exc:
                interrupted = exc
                if cleanup.done():
                    break
            except Exception as exc:
                if original is None:
                    raise
                original.add_note(f"Creator scorer workspace cleanup failed: {exc}")
                break
        # A cancellation can arrive at the same instant cleanup finishes.
        # Retrieve its exception even when shield raised cancellation first.
        if cleanup.done() and not cleanup.cancelled():
            failure = cleanup.exception()
            if failure is not None and original is None:
                raise failure
            if failure is not None and original is not None:
                note = f"Creator scorer workspace cleanup failed: {failure}"
                if note not in getattr(original, "__notes__", []):
                    original.add_note(note)
        if interrupted is not None and original is None:
            raise interrupted
