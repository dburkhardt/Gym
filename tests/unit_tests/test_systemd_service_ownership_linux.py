# SPDX-License-Identifier: Apache-2.0
"""Opt-in real Linux/systemd user-scope control. No model/server calls."""

import json
import os
import subprocess
import sys
import time

import psutil
import pytest

from nemo_gym.cli.service_ownership import ServiceOwnership


pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or os.environ.get("NEMO_GYM_TEST_SYSTEMD") != "1",
    reason="Requires explicit existing Linux user-manager qualification",
)


def wait_for(check, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = check()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("bounded control condition did not become true")


def alive(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_real_scope_command_environment_cwd_stdio_and_failure(tmp_path):
    owner = ServiceOwnership(tmp_path / "ownership", {"source": "real-linux-v1"}, mode="systemd")
    env = {**os.environ, "TEST_PRIVATE_ENV_VALUE": "fixture-only-not-a-credential"}
    command = owner.command(
        'read line; test "$TEST_PRIVATE_ENV_VALUE" = fixture-only-not-a-credential; printf \'%s:%s\' "$PWD" "$line"; exit 7',
        env,
    )
    try:
        p = subprocess.run(
            command,
            cwd=tmp_path,
            env=env,
            input="original-stdin\n",
            capture_output=True,
            text=True,
            start_new_session=True,
            timeout=15,
        )
        assert p.returncode == 7, (p.returncode, p.stderr)
        assert p.stdout == str(tmp_path) + ":original-stdin"
        records = [json.loads(x.read_text()) for x in (tmp_path / "ownership").glob("*.json")]
        assert records[0]["systemd"]["invocation_id"]
        assert "fixture-only-not-a-credential" not in json.dumps(records)
    finally:
        owner.close()


def test_real_coordinator_sigkill_fresh_reconcile_detached_child_and_sentinel(tmp_path):
    directory = tmp_path / "ownership"
    ready = tmp_path / "ready.json"
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    child = "import subprocess,sys,time,json,os; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); Path(sys.argv[1]).write_text(json.dumps({'root':os.getpid(),'detached':p.pid})); time.sleep(60)"
    import shlex

    shell = shlex.join([sys.executable, "-c", child, str(ready)])
    coordinator_code = """import os,subprocess,sys,time
from nemo_gym.cli.service_ownership import ServiceOwnership
owner=ServiceOwnership(sys.argv[1],{'source':'real-linux-v1'},mode='systemd')
env=os.environ.copy();command=owner.command(sys.argv[2],env)
process=subprocess.Popen(command,env=env,start_new_session=True)
process.wait()
owner.close()
"""
    coordinator = subprocess.Popen([sys.executable, "-c", coordinator_code, str(directory), shell])
    original = None
    path = None
    pids = None
    try:
        wait_for(ready.exists)
        pids = json.loads(ready.read_text())
        path = next(directory.glob("*.json"))
        original = path.read_bytes()
        value = json.loads(original)
        assert value["systemd"]["invocation_id"]
        coordinator.kill()
        coordinator.wait(timeout=5)
        assert alive(pids["detached"]) and sentinel.poll() is None
        bad = json.loads(original)
        bad["systemd"]["invocation_id"] = "0" * 32
        path.write_text(json.dumps(bad))
        with pytest.raises(RuntimeError, match="invocation changed"):
            ServiceOwnership(directory, {"source": "real-linux-v1"}, mode="systemd")
        assert alive(pids["detached"]) and sentinel.poll() is None
        path.write_bytes(original)
        fresh = "from nemo_gym.cli.service_ownership import ServiceOwnership; import sys; o=ServiceOwnership(sys.argv[1],{'source':'real-linux-v1'},mode='systemd');o.close()"
        result = subprocess.run(
            [sys.executable, "-c", fresh, str(directory)], capture_output=True, text=True, timeout=20
        )
        assert result.returncode == 0, result.stderr
        wait_for(lambda: not alive(pids["root"]) and not alive(pids["detached"]))
        assert json.loads(path.read_text())["cleanup_confirmed"] is True
        assert sentinel.poll() is None
    finally:
        if coordinator.poll() is None:
            coordinator.kill()
            coordinator.wait(timeout=5)
        try:
            if path and original:
                path.write_bytes(original)
            if directory.exists():
                owner = ServiceOwnership(directory, {"source": "real-linux-v1"}, mode="systemd")
                owner.close()
        finally:
            sentinel.terminate()
            sentinel.wait(timeout=5)
