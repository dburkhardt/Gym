# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real child timeout with fake Docker creation: no scientific/model execution."""

import json
import os
import shlex
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from benchmarks.scivisagentbench import managed
from benchmarks.scivisagentbench.test_managed import tree


@pytest.mark.asyncio
async def test_timeout_reaps_creator_before_bridge_and_viewer_cleanup(tmp_path, monkeypatch):
    source = tree(tmp_path / "source")
    data = tree(tmp_path / "data")
    (tmp_path / "source" / "src").mkdir()
    (tmp_path / "source" / "file").rename(tmp_path / "source" / "src" / "file")
    source["sha256"] = {"src/file": source["sha256"]["file"]}
    packet = {
        "image": "sha256:" + "a" * 64,
        "mounts": [{**source, "target": "/source"}, {**data, "target": "/workspace/data"}],
        "viewer_command": ["fixture"],
        "napari_mcp_command": ["fixture"],
        "filesystem_mcp_command": ["fixture"],
        "reference_root": "unused",
        "reference_sha256": {},
    }
    (tmp_path / "packet.json").write_text(json.dumps({"cases": {f"case_{i}": packet for i in range(1, 12)}}))
    (tmp_path / "native.json").write_text("{}")
    config = SimpleNamespace(
        runtime_packet=str(tmp_path / "packet.json"),
        native_python=sys.executable,
        native_config=str(tmp_path / "native.json"),
        source_root=str(tmp_path / "source"),
        max_tokens=16384,
        timeout_s=0.2,
    )
    output = tmp_path / "out"
    pidfile = tmp_path / "pid"
    events = []

    class Sandbox:
        def __init__(self, *args):
            pass

        async def start(self):
            events.append("start")

        async def serialize(self):
            return {"sandbox_id": "owned-fixture"}

        async def stop(self):
            events.append("stop")

        async def exec(self, *args, **kwargs):
            (output / "workspace" / "viewer-ready.json").write_text("{}")
            return SimpleNamespace(return_code=0)

    monkeypatch.setattr(managed, "AsyncSandbox", Sandbox)
    monkeypatch.setattr(managed, "validate_source", lambda *args: {})
    original_exec = managed.LocalProvider.exec

    async def short_exec(self, handle, command, **kwargs):
        kwargs["timeout_s"] = 1
        return await original_exec(self, handle, command, **kwargs)

    monkeypatch.setattr(managed.LocalProvider, "exec", short_exec)
    original_join = shlex.join

    def command(args):
        if "-m" in args:
            script = f"import os,time;open({str(pidfile)!r},'w').write(str(os.getpid()));time.sleep(30)"
            return original_join([sys.executable, "-c", script])
        return original_join(args)

    monkeypatch.setattr(managed.shlex, "join", command)

    @asynccontextmanager
    async def bridge(*args, **kwargs):
        try:
            yield
        finally:
            pid = int(pidfile.read_text())
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
            events.append("bridge-settled")

    monkeypatch.setattr(managed, "bridge", bridge)
    with pytest.raises(RuntimeError, match="process failed"):
        await managed.execute(config, {"case_name": "case_1", "source_case": {"assert": []}}, output, "unused")
    assert events == ["start", "bridge-settled", "stop"]
    assert json.loads((output / "viewer-lifecycle.json").read_text())["cleanup_confirmed"]
