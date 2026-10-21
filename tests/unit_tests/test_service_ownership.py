# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import sys
import time
import uuid

import psutil
import pytest

from nemo_gym.cli.service_ownership import ServiceOwnership, host_identity, identity


def receipt(directory, **changes):
    value = {
        "version": 1,
        "binding": {"source": "frozen"},
        "host": host_identity(),
        "coordinator": {"pid": 99999999, "created": time.time() - 1},
        "root": {"pid": 99999998, "created": 0},
        "token": uuid.uuid4().hex,
        "command_sha256": "a" * 64,
        "cleanup_confirmed": False,
    }
    value.update(changes)
    path = directory / "old.json"
    path.write_text(json.dumps(value))
    return path


@pytest.mark.parametrize(
    "change",
    [{"binding": {}}, {"host": {}}, {"version": 2}, {"coordinator": identity(psutil.Process())}],
)
def test_invalid_or_active_receipt_refused_before_signals(tmp_path, monkeypatch, change):
    path = receipt(tmp_path, **change)
    before = path.read_bytes()
    monkeypatch.setattr(psutil.Process, "send_signal", lambda *_: pytest.fail("must not signal"))
    with pytest.raises(RuntimeError):
        ServiceOwnership(tmp_path, {"source": "frozen"})
    assert path.read_bytes() == before


def test_lease_and_symlink_refusal(tmp_path):
    owner = ServiceOwnership(tmp_path / "owned", {})
    try:
        with pytest.raises(BlockingIOError):
            ServiceOwnership(tmp_path / "owned", {})
        (tmp_path / "alias").symlink_to(tmp_path / "owned")
        with pytest.raises(RuntimeError, match="symlinks"):
            ServiceOwnership(tmp_path / "alias", {})
    finally:
        owner.close()


def test_stale_pid_cannot_kill_unrelated_group(tmp_path):
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        actual = psutil.Process(sentinel.pid)
        receipt(tmp_path, root={"pid": sentinel.pid, "created": actual.create_time() - 1})
        owner = ServiceOwnership(tmp_path, {"source": "frozen"})
        owner.close()
        assert sentinel.poll() is None
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=5)


def test_missing_old_root_settles_without_signaling(tmp_path, monkeypatch):
    path = receipt(tmp_path)
    monkeypatch.setattr(psutil.Process, "send_signal", lambda *_: pytest.fail("must not signal"))
    owner = ServiceOwnership(tmp_path, {"source": "frozen"})
    owner.close()
    assert json.loads(path.read_text())["cleanup_confirmed"] is True


def test_pending_launch_without_process_settles(tmp_path):
    path = receipt(tmp_path, root=None)
    owner = ServiceOwnership(tmp_path, {"source": "frozen"})
    owner.close()
    assert json.loads(path.read_text())["cleanup_confirmed"]


def test_detached_token_child_is_settled(tmp_path):
    token = uuid.uuid4().hex
    from nemo_gym.cli.service_ownership import TOKEN

    env = {**os.environ, TOKEN: token}
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], env=env, start_new_session=True)
    try:
        receipt(tmp_path, token=token, root=None)
        owner = ServiceOwnership(tmp_path, {"source": "frozen"})
        owner.close()
        child.wait(timeout=2)
        assert not psutil.pid_exists(child.pid)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=2)


def test_only_declared_deployment_fields_excluded():
    from copy import deepcopy

    from nemo_gym.cli.service_ownership import component_binding

    config = {
        "agent": {
            "responses_api_agents": {
                "simple_agent": {
                    "host": "localhost",
                    "port": 1,
                    "task": {"host": "science", "port": 42, "num_workers": 3},
                }
            }
        }
    }
    base = component_binding(config)
    deployed = deepcopy(config)
    deployed["agent"]["responses_api_agents"]["simple_agent"]["port"] = 2
    assert component_binding(deployed) == base
    deployed["agent"]["responses_api_agents"]["simple_agent"]["task"]["port"] = 43
    assert component_binding(deployed) != base


def test_startup_failure_releases_ownership_lease(tmp_path, monkeypatch):
    from nemo_gym.cli.env import RunHelper

    runner = RunHelper()
    runner._service_ownership = ServiceOwnership(tmp_path, {})

    def fail_start(self, config):
        raise RuntimeError("fixture startup failure")

    monkeypatch.setattr(RunHelper, "_start", fail_start)
    with pytest.raises(RuntimeError, match="fixture startup failure"):
        runner.start(None)
    owner = ServiceOwnership(tmp_path, {})
    owner.close()


def test_confirmed_closed_receipt_allows_same_process_reentry(tmp_path):
    receipt(tmp_path, coordinator=identity(psutil.Process()), cleanup_confirmed=True)
    owner = ServiceOwnership(tmp_path, {"source": "frozen"})
    owner.close()


def test_nonfinite_identity_refused(tmp_path):
    receipt(tmp_path, coordinator={"pid": 99999999, "created": float("nan")})
    with pytest.raises(RuntimeError, match="Invalid service process identity"):
        ServiceOwnership(tmp_path, {"source": "frozen"})


@pytest.mark.parametrize("after_denial", ["gone", "zombie", "running", "denied"])
def test_environment_access_race_requires_observed_termination(monkeypatch, after_denial):
    from types import SimpleNamespace

    from nemo_gym.cli.service_ownership import TOKEN, owned_members

    class RacingProcess:
        denied = False

        def create_time(self):
            return 10

        def uids(self):
            return SimpleNamespace(real=os.getuid())

        def status(self):
            if self.denied:
                if after_denial == "gone":
                    raise psutil.NoSuchProcess(123)
                if after_denial == "denied":
                    raise psutil.AccessDenied(123)
                if after_denial == "zombie":
                    return psutil.STATUS_ZOMBIE
            return psutil.STATUS_RUNNING

        def environ(self):
            self.denied = True
            raise psutil.AccessDenied(123)

    owned = SimpleNamespace(
        create_time=lambda: 10,
        uids=lambda: SimpleNamespace(real=os.getuid()),
        status=lambda: psutil.STATUS_RUNNING,
        environ=lambda: {TOKEN: "owned"},
    )
    monkeypatch.setattr(psutil, "process_iter", lambda: iter([RacingProcess(), owned]))
    if after_denial in {"gone", "zombie"}:
        assert owned_members("owned", 0) == [owned]
    else:
        with pytest.raises(psutil.AccessDenied):
            owned_members("owned", 0)
