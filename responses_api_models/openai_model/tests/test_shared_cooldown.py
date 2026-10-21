# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import multiprocessing
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientResponseError, ClientSession

from nemo_gym.server_utils import ServerClient
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig
from responses_api_models.openai_model.shared_cooldown import SharedCooldown


def config(directory=None, namespace=None):
    return SimpleModelServerConfig(
        host="127.0.0.1",
        port=1,
        name="fixture",
        entrypoint="",
        openai_base_url="http://fixture.invalid/v1",
        openai_api_key="SECRET-FIXTURE",
        openai_model="fixture",
        max_concurrent_requests=1,
        upstream_max_num_tries=1,
        max_http_attempts=1,
        upstream_rate_limit_cooldown_seconds=0.1,
        upstream_retry_policy={"max_attempts": 1},
        upstream_rate_limit_shared_directory=directory,
        upstream_rate_limit_shared_namespace=namespace,
    )


def server(directory, namespace="fixture"):
    return SimpleModelServer(
        config=config(directory, namespace), server_client=MagicMock(spec=ServerClient, global_config_dict={})
    )


def process_request(url, directory, output, ready, release):
    async def run():
        model = server(Path(directory))
        ready.set()
        assert await asyncio.to_thread(release.wait, 15)
        before_call = time.monotonic()
        async with ClientSession() as client:

            async def operation():
                async with client.post(url) as response:
                    response.raise_for_status()
                    return await response.text()

            try:
                await model._call_upstream(operation)
                output.put(("ok", before_call))
            except ClientResponseError as error:
                output.put((error.status, before_call))

    asyncio.run(run())


def test_actual_separate_processes_observe_one_scope(tmp_path):
    """An already-initialized peer must wait on another process's persisted 429."""
    arrivals = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            arrivals.append(time.monotonic())
            self.send_response(429 if len(arrivals) == 1 else 200)
            if len(arrivals) == 1:
                self.send_header("Retry-After", "3")
            self.end_headers()
            self.wfile.write(b"fixture")

        def log_message(self, *args):
            pass

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    ctx = multiprocessing.get_context("spawn")
    output = ctx.Queue()
    children = []
    try:
        releases = [ctx.Event(), ctx.Event()]
        readies = [ctx.Event(), ctx.Event()]
        for index in range(2):
            child = ctx.Process(
                target=process_request,
                args=(
                    f"http://127.0.0.1:{provider.server_port}/",
                    str(tmp_path),
                    output,
                    readies[index],
                    releases[index],
                ),
            )
            children.append(child)
            child.start()
        assert all(ready.wait(15) for ready in readies)
        releases[0].set()
        assert output.get(timeout=15)[0] == 429
        # Both model objects are already initialized. Release the peer while the
        # persisted deadline is still active; imports cannot create the wait.
        state = json.loads(next(tmp_path.glob("*.json")).read_text())
        remaining_at_release = state["until_monotonic"] - time.monotonic()
        assert remaining_at_release > 2.5
        releases[1].set()
        result, before_call = output.get(timeout=15)
        assert result == "ok"
        assert before_call < state["until_monotonic"] - 2.4
        assert arrivals[1] >= state["until_monotonic"] - 0.02
        for child in children:
            child.join(timeout=5)
            assert child.exitcode == 0
        assert len(arrivals) == 2
        assert arrivals[1] - arrivals[0] >= 2.95
        (tmp_path / "loopback-proof.json").write_text(
            json.dumps(
                {
                    "worker_exit_codes": [child.exitcode for child in children],
                    "physical_loopback_requests": len(arrivals),
                    "retry_after_seconds": 3,
                    "observed_interarrival_seconds": arrivals[1] - arrivals[0],
                    "both_workers_initialized_before_first_rejection": True,
                    "second_call_start_before_deadline_seconds": state["until_monotonic"] - before_call,
                    "actual_second_call_wait_seconds": arrivals[1] - before_call,
                    "external_provider_calls": 0,
                },
                indent=2,
            )
        )
        state = next(p for p in tmp_path.glob("*.json") if p.name != "loopback-proof.json").read_text()
        assert "SECRET-FIXTURE" not in state and "fixture.invalid" not in state
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(5)
        provider.shutdown()
        provider.server_close()
        thread.join(5)


def test_scope_monotonic_restart_and_corruption(tmp_path, monkeypatch):
    first = SharedCooldown(tmp_path, "scope", "http://fixture.invalid/v1", "model")
    first.extend(10)
    deadline = json.loads(first.path.read_text())["until_monotonic"]
    first.extend(0)
    assert json.loads(first.path.read_text())["until_monotonic"] == deadline
    restarted = SharedCooldown(tmp_path, "scope", "http://fixture.invalid/v1", "model")
    monkeypatch.setattr(time, "time", lambda: 1e15)
    assert restarted.remaining() > 9
    independent = SharedCooldown(tmp_path, "other", "http://fixture.invalid/v1", "model")
    assert independent.remaining() == 0
    with pytest.raises(ValueError, match="different endpoint/model"):
        SharedCooldown(tmp_path, "scope", "http://fixture.invalid/v1", "different")
    restarted.boot = "0" * 64
    assert restarted.remaining() == 0
    first.path.write_text("{")
    with pytest.raises(ValueError):
        first.remaining()


def test_atomic_failure_preserves_prior_state(tmp_path, monkeypatch):
    from responses_api_models.openai_model import shared_cooldown

    state = SharedCooldown(tmp_path, "fixture", "http://fixture.invalid", "model")
    state.extend(1)
    before = state.path.read_bytes()
    monkeypatch.setattr(
        shared_cooldown.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("fixture publication failure"))
    )
    with pytest.raises(OSError):
        state.extend(10)
    assert state.path.read_bytes() == before
    assert not list(tmp_path.glob(".cooldown-*"))


def test_invalid_config_and_secret_url(tmp_path):
    with pytest.raises(ValueError, match="both directory"):
        config(tmp_path, None)
    with pytest.raises(ValueError, match="userinfo"):
        SharedCooldown(tmp_path, "scope", "http://user:secret@host/v1", "model")
    with pytest.raises(ValueError, match="userinfo"):
        SharedCooldown(tmp_path, "scope", "http://host/v1?key=secret", "model")


@pytest.mark.asyncio
async def test_wait_cancellation_releases_slot_and_never_sends(tmp_path):
    model = server(tmp_path)
    model._shared_cooldown.extend(10)
    calls = []

    async def operation():
        calls.append(1)

    task = asyncio.create_task(model._call_upstream(operation))
    await asyncio.sleep(0.03)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not calls and not model._semaphore.locked()
    peer = server(tmp_path)
    assert await peer._cooldown_remaining() > 9


@pytest.mark.asyncio
async def test_unknown_is_not_retried_in_shared_mode(tmp_path):
    model = server(tmp_path)
    before = model._shared_cooldown.path.read_bytes()
    calls = []

    async def operation():
        calls.append(1)
        raise TimeoutError("accepted unknown fixture")

    with pytest.raises(TimeoutError):
        await model._call_upstream(operation)
    assert calls == [1] and model._shared_cooldown.path.read_bytes() == before


def hold_lock(path, ready, release):
    import fcntl

    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        ready.set()
        release.wait(10)


@pytest.mark.asyncio
async def test_real_process_held_lock_cancellation_settles(tmp_path):
    model = server(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    child = ctx.Process(target=hold_lock, args=(str(model._shared_cooldown.lock_path), ready, release))
    child.start()
    try:
        assert await asyncio.to_thread(ready.wait, 10)
        calls = []

        async def operation():
            calls.append(1)

        started = time.monotonic()
        task = asyncio.create_task(model._call_upstream(operation))
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as error:
            await task
        elapsed = time.monotonic() - started
        assert 0.9 <= elapsed < 3
        assert "TimeoutError" in str(error.value.__notes__)
        assert not calls and not model._semaphore.locked()
        # Synchronous startup is also bounded, rather than waiting forever.
        before = time.monotonic()
        with pytest.raises(TimeoutError, match="one second"):
            server(tmp_path)
        assert time.monotonic() - before < 3
    finally:
        release.set()
        child.join(5)
        if child.is_alive():
            child.terminate()
            child.join(5)
        assert child.exitcode == 0


def test_nonobject_state_fails_closed(tmp_path):
    state = SharedCooldown(tmp_path, "scope", "http://fixture.invalid", "model")
    state.path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        state.remaining()


def construct_scope(directory, model, ready, release, output):
    ready.set()
    if not release.wait(10):
        raise RuntimeError("Fixture barrier timed out")
    try:
        SharedCooldown(Path(directory), "same-namespace", "http://fixture.invalid", model)
        output.put("bound")
    except ValueError:
        output.put("rejected")


def test_concurrent_conflicting_constructors_bind_before_dispatch(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    release, output = ctx.Event(), ctx.Queue()
    readies = [ctx.Event(), ctx.Event()]
    children = [
        ctx.Process(target=construct_scope, args=(str(tmp_path), model, ready, release, output))
        for model, ready in zip(["model-a", "model-b"], readies, strict=True)
    ]
    try:
        for child in children:
            child.start()
        assert all(ready.wait(10) for ready in readies)
        release.set()
        assert sorted([output.get(timeout=10), output.get(timeout=10)]) == ["bound", "rejected"]
        for child in children:
            child.join(5)
            assert child.exitcode == 0
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(5)
