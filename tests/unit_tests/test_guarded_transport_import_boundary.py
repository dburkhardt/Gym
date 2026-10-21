# SPDX-License-Identifier: Apache-2.0
"""Minimal creator runtimes need only httpx, including standalone copies."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_legacy_exports_are_canonical():
    from nemo_gym import guarded_chat_transport as canonical
    from nemo_gym.sandbox import guarded_chat_transport as legacy

    for name in ("GuardedChatTransport", "GuardedSyncChatTransport", "publish"):
        assert getattr(legacy, name) is getattr(canonical, name)


@pytest.mark.parametrize("standalone", [False, True])
def test_restricted_child_import_and_standalone_copy(tmp_path, standalone):
    source = Path(__file__).resolve().parents[2]
    shutil.copyfile(source / "nemo_gym/guarded_chat_transport.py", tmp_path / "guarded_chat_transport.py")
    module = "guarded_chat_transport" if standalone else "nemo_gym.guarded_chat_transport"
    script = """
import importlib, importlib.abc, pathlib, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ("omegaconf", "ray", "harbor", "hydra", "nemo_gym.sandbox", "nemo_gym.telemetry")
        if sys.argv[2] == "True":
            blocked += ("nemo_gym",)
        if any(fullname == p or fullname.startswith(p + ".") for p in blocked):
            raise ModuleNotFoundError("restricted dependency: " + fullname)
sys.meta_path.insert(0, Block())
module = importlib.import_module(sys.argv[1])
import httpx
calls = []
async def provider(request):
    calls.append(request)
    return httpx.Response(200, json={"model": "model", "choices": [{"finish_reason": "stop"}]})
transport = module.GuardedSyncChatTransport(socket_path="unused", token="local", lineage="attempt",
    digest="config", namespace="control", model="model", output=pathlib.Path("calls"),
    transport=httpx.MockTransport(provider))
with httpx.Client(transport=transport) as client:
    assert client.post("http://local/v1/chat/completions", json={"model": "model"}).status_code == 200
assert len(calls) == 1
assert pathlib.Path("calls/000000.response.json").is_file()
"""
    env = dict(os.environ, PYTHONPATH=str(tmp_path if standalone else source))
    result = subprocess.run(
        [sys.executable, "-c", script, module, str(standalone)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
