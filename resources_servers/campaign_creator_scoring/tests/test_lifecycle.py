# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from pathlib import Path

import psutil
import pytest

from resources_servers.campaign_creator_scoring import app


@pytest.fixture
def worker(tmp_path, monkeypatch):
    directory = tmp_path / "server"
    directory.mkdir()
    monkeypatch.setattr(app, "__file__", str(directory / "app.py"))
    # Shared LocalProvider's semaphore is per service loop; these tests use new loops.
    monkeypatch.setattr(app.LocalProvider, "_semaphores", {})
    source = tmp_path.parent / "benchmarks/scientific_agent_campaign/deterministic.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# source fixture\n")
    return directory / "score_worker.py"


async def wait_file(path):
    for _ in range(200):
        if path.exists():
            return
        await asyncio.sleep(0.01)
    pytest.fail("worker did not start")


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_and_cancel_kill_scientific_child(worker, tmp_path, cancel):
    marker = tmp_path / "pids.json"
    worker.write_text("""import json, os, subprocess, sys, time
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
Path(r['answer']).write_text(json.dumps([os.getpid(),p.pid,str(Path(sys.argv[1]).parent)]))
time.sleep(60)
""")
    task = asyncio.create_task(
        app.run_score(
            benchmark="control", upstream_repo=tmp_path, task={}, answer=str(marker), timeout_s=2 if not cancel else 30
        )
    )
    await wait_file(marker)
    parent, child, workspace = json.loads(marker.read_text())
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    for pid in [parent, child]:
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    assert not Path(workspace).exists()


@pytest.mark.asyncio
async def test_failure_is_not_score_zero_and_workspace_removed(worker, tmp_path):
    marker = tmp_path / "workspace"
    worker.write_text("""import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
Path(r['answer']).write_text(str(Path(sys.argv[1]).parent))
raise ValueError('creator scorer control failure')
""")
    with pytest.raises(RuntimeError, match="creator scorer control failure"):
        await app.run_score(benchmark="control", upstream_repo=tmp_path, task={}, answer=str(marker), timeout_s=10)
    assert not Path(marker.read_text()).exists()


@pytest.mark.asyncio
async def test_worker_preserves_signal_main_thread_and_result(worker, tmp_path):
    worker.write_text("""import json,signal,sys,threading
from pathlib import Path
assert threading.current_thread() is threading.main_thread()
signal.signal(signal.SIGALRM, lambda *a: None)
source=Path(__file__).resolve().parents[2]/'benchmarks/scientific_agent_campaign/deterministic.py'
import hashlib
Path(sys.argv[2]).write_text(json.dumps({'details':{'reward':0.5,'metric':'native-control'},'runtime':{'source_path':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}}))
""")
    result = await app.run_score(benchmark="control", upstream_repo=tmp_path, task={}, answer="", timeout_s=10)
    assert result == {"reward": 0.5, "metric": "native-control"}


@pytest.mark.asyncio
async def test_two_process_bound(worker, tmp_path):
    worker.write_text("""import json,sys,time
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
Path(r['answer']).write_text('started')
time.sleep(.5)
source=Path(__file__).resolve().parents[2]/'benchmarks/scientific_agent_campaign/deterministic.py'
import hashlib
Path(sys.argv[2]).write_text(json.dumps({'details':{'reward':1},'runtime':{'source_path':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}}))
""")
    markers = [tmp_path / str(i) for i in range(3)]
    calls = [
        asyncio.create_task(
            app.run_score(benchmark="control", upstream_repo=tmp_path, task={}, answer=str(p), timeout_s=10)
        )
        for p in markers
    ]
    await wait_file(markers[0])
    await wait_file(markers[1])
    assert not markers[2].exists()
    assert [r["reward"] for r in await asyncio.gather(*calls)] == [1, 1, 1]


@pytest.mark.asyncio
async def test_cancel_during_spawn_settles_child_before_return(worker, tmp_path, monkeypatch):
    marker = tmp_path / "pids.json"
    worker.write_text("import os,time; time.sleep(60)")
    original = asyncio.create_subprocess_exec
    spawned = asyncio.Event()
    release = asyncio.Event()
    processes = []

    async def delayed_spawn(*args, **kwargs):
        process = await original(*args, **kwargs)
        processes.append(process)
        spawned.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    call = asyncio.create_task(
        app.run_score(benchmark="control", upstream_repo=tmp_path, task={}, answer=str(marker), timeout_s=30)
    )
    await spawned.wait()
    call.cancel()
    await asyncio.sleep(0.01)
    call.cancel()
    assert not call.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert processes[0].returncode is not None


@pytest.mark.asyncio
async def test_cleanup_failure_preserves_scorer_error(worker, tmp_path, monkeypatch):
    worker.write_text("raise ValueError('original scoring failure')")
    close = app.LocalProvider.close

    async def failing_close(self, handle):
        await close(self, handle)
        raise OSError("cleanup control failure")

    monkeypatch.setattr(app.LocalProvider, "close", failing_close)
    with pytest.raises(RuntimeError, match="original scoring failure") as error:
        await app.run_score(benchmark="control", upstream_repo=tmp_path, task={}, answer="", timeout_s=10)
    assert "cleanup control failure" in error.value.__notes__[0]


@pytest.mark.asyncio
async def test_repeated_cancel_settles_workspace_cleanup(worker, tmp_path, monkeypatch):
    worker.write_text("import time; time.sleep(60)")
    close = app.LocalProvider.close
    started = asyncio.Event()
    release = asyncio.Event()
    workspaces = []

    async def slow_close(self, handle):
        workspaces.append(handle.raw["workspace"])
        started.set()
        await release.wait()
        await close(self, handle)

    monkeypatch.setattr(app.LocalProvider, "close", slow_close)
    call = asyncio.create_task(
        app.run_score(benchmark="control", upstream_repo=tmp_path, task={}, answer="", timeout_s=0.1)
    )
    await started.wait()
    call.cancel()
    await asyncio.sleep(0.01)
    call.cancel()
    assert not call.done()
    release.set()
    with pytest.raises(TimeoutError):
        await call
    assert not workspaces[0].exists()


@pytest.mark.asyncio
async def test_source_identity_mismatch_rejects_score(worker, tmp_path):
    worker.write_text(
        "import json,sys; from pathlib import Path; Path(sys.argv[2]).write_text(json.dumps({'details':{'reward':1},'runtime':{'source_path':'stale'}}))"
    )
    with pytest.raises(RuntimeError, match="different source"):
        await app.run_score(benchmark="control", upstream_repo=tmp_path, task={}, answer="", timeout_s=10)


@pytest.mark.asyncio
async def test_simultaneous_cleanup_failure_cancel_keeps_original(worker, tmp_path, monkeypatch):
    worker.write_text("raise ValueError('original scoring failure')")
    close = app.LocalProvider.close
    caller = asyncio.current_task()

    async def failing_close(self, handle):
        await close(self, handle)
        caller.cancel()
        raise OSError("simultaneous cleanup failure")

    monkeypatch.setattr(app.LocalProvider, "close", failing_close)
    with pytest.raises(RuntimeError, match="original scoring failure") as error:
        await app.run_score(benchmark="control", upstream_repo=tmp_path, task={}, answer="", timeout_s=10)
    assert "simultaneous cleanup failure" in error.value.__notes__[0]
