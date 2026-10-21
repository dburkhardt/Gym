# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual repeated cancellation must settle the owned native resume request."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from nemo_gym._checkpoint import evaluation


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_repeated_cancel_settles_resume_preserving_primary(monkeypatch, tmp_path, cleanup_fails):
    controller = evaluation.EvaluationCheckpoint(
        directory=tmp_path, client=None, config={}, binding={}, capture_dirs=[], artifacts=[]
    )
    controller.participants = object()
    started = asyncio.Event()
    finish = asyncio.Event()
    settled = asyncio.Event()
    original = ValueError("original prepare failure")
    controller.discover = AsyncMock(side_effect=original)

    async def resume(*args, **kwargs):
        started.set()
        await finish.wait()
        settled.set()
        if cleanup_fails:
            raise RuntimeError("resume failed")

    monkeypatch.setattr(evaluation.coordination, "resume", resume)
    task = asyncio.create_task(controller.checkpoint())
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and not controller.admission.is_set()
    finish.set()
    with pytest.raises(ValueError, match="original prepare failure") as caught:
        await task
    assert caught.value is original
    assert settled.is_set() and controller.admission.is_set()
    assert original.__notes__
