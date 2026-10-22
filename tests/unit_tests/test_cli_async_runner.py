# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nemo_gym.cli.async_runner import run_with_sigterm_cleanup


def test_completion_and_handler_restoration():
    previous = signal.getsignal(signal.SIGTERM)

    async def complete():
        return 17

    assert run_with_sigterm_cleanup(complete()) == 17
    assert signal.getsignal(signal.SIGTERM) == previous


def test_unrelated_cancellation_stays_cancellation():
    async def cancel():
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        run_with_sigterm_cleanup(cancel())


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process signals")
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_signal_settles_owned_detached_child(tmp_path: Path, signum: int):
    script = """
import asyncio, subprocess, sys, time
from pathlib import Path
from nemo_gym.cli.async_runner import sigterm_cleanup_scope
p=Path(sys.argv[1])
async def work():
    child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"],start_new_session=True)
    (p/"ready").write_text(str(child.pid))
    try:
        await asyncio.sleep(60)
    finally:
        (p/"cleaning").write_text("yes")
        await asyncio.sleep(0.3)
        child.terminate()
        child.wait(timeout=5)
        (p/"cleaned").write_text("yes")
with sigterm_cleanup_scope() as run:
    try:
        run(work())
    finally:
        (p/"outer-start").write_text("yes")
        time.sleep(0.3)
        (p/"outer-finally").write_text("yes")
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)])
    child_pid = None
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (tmp_path / "ready").exists()
        child_pid = int((tmp_path / "ready").read_text())
        process.send_signal(signum)
        if signum == signal.SIGTERM:
            while not (tmp_path / "cleaning").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert (tmp_path / "cleaning").exists()
            process.send_signal(signal.SIGTERM)  # Must not interrupt cleanup again.
            while not (tmp_path / "outer-start").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert (tmp_path / "outer-start").exists()
            process.send_signal(signal.SIGTERM)  # Also protect synchronous service shutdown.
        exit_code = process.wait(timeout=10)
        assert exit_code == 143 if signum == signal.SIGTERM else exit_code != 0
        assert (tmp_path / "cleaned").exists()
        assert (tmp_path / "outer-finally").exists()
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_swallowed_sigterm_cancellation_cannot_succeed():
    async def swallow():
        os.kill(os.getpid(), signal.SIGTERM)
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            return "incorrect success"

    with pytest.raises(SystemExit) as error:
        run_with_sigterm_cleanup(swallow())
    assert error.value.code == 143


def test_startup_scope_restores_handler_on_error():
    from nemo_gym.cli.async_runner import sigterm_startup_scope

    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(ValueError, match="startup failure"):
        with sigterm_startup_scope():
            raise ValueError("startup failure")
    assert signal.getsignal(signal.SIGTERM) == previous
    with sigterm_startup_scope():
        pass
    assert signal.getsignal(signal.SIGTERM) == previous


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process signals")
@pytest.mark.parametrize("ordinary_failure", [False, True])
def test_actual_runhelper_start_settles_service_on_sigterm(tmp_path: Path, ordinary_failure: bool):
    script = """
import subprocess,sys,time,types
from pathlib import Path
from nemo_gym.cli.env import RunHelper
p=Path(sys.argv[1])
helper=RunHelper.__new__(RunHelper)
helper._startup=None
child=None
def start(self, config):
    global child
    child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"],start_new_session=True)
    (p/"ready").write_text(str(child.pid))
    if sys.argv[2] == "failure":
        raise ValueError("ordinary startup failure")
    time.sleep(60)
def shutdown(self):
    (p/"cleaning").write_text("yes")
    time.sleep(.4)
    child.terminate()
    child.wait(timeout=5)
    (p/"cleaned").write_text("yes")
helper._start=types.MethodType(start,helper)
helper.shutdown=types.MethodType(shutdown,helper)
helper.start(None)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path), "failure" if ordinary_failure else "wait"]
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 30
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (tmp_path / "ready").exists()
        child_pid = int((tmp_path / "ready").read_text())
        if not ordinary_failure:
            process.send_signal(signal.SIGTERM)
        while not (tmp_path / "cleaning").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (tmp_path / "cleaning").exists()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) == 143
        assert (tmp_path / "cleaned").exists()
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
