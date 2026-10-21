# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys

import pytest

from nemo_gym.cli.port_ownership import PortOwnership


def test_aliases_and_fixed_head_conflict(tmp_path):
    root = tmp_path / "locks"
    owner = PortOwnership([("head", "0.0.0.0", 20001)], root)
    try:
        for host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            with pytest.raises(RuntimeError, match="owns local port"):
                PortOwnership([("component", host, 20001)], root)
        other = PortOwnership([("component", "localhost", 20002)], root)
        other.close()
    finally:
        owner.close()
    PortOwnership([("head", "localhost", 20001)], root).close()


def test_duplicate_and_partial_acquisition_release(tmp_path):
    with pytest.raises(RuntimeError, match="overlap"):
        PortOwnership([("head", "localhost", 20001), ("component", "0.0.0.0", 20001)], tmp_path / "locks")
    owner = PortOwnership([("existing", "localhost", 20002)], tmp_path / "locks")
    with pytest.raises(RuntimeError, match="owns local port"):
        PortOwnership([("first", "localhost", 20001), ("second", "localhost", 20002)], tmp_path / "locks")
    PortOwnership([("recovered", "localhost", 20001)], tmp_path / "locks").close()
    owner.close()


@pytest.mark.parametrize("terminal", ["release", "crash"])
def test_actual_two_process_ownership_and_terminal_release(tmp_path, terminal):
    root = tmp_path / "locks"
    code = """import sys
from nemo_gym.cli.port_ownership import PortOwnership
owner=PortOwnership([("head","0.0.0.0",20001)],sys.argv[1])
print("ready",flush=True)
sys.stdin.readline()
owner.close()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(root)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        with pytest.raises(RuntimeError, match="owns local port"):
            PortOwnership([("second", "localhost", 20001)], root)
        assert process.poll() is None  # A collision never kills the owner.
        if terminal == "crash":
            process.kill()
        else:
            process.stdin.write("done\n")
            process.stdin.flush()
        process.wait(timeout=10)
        PortOwnership([("next", "localhost", 20001)], root).close()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def test_symlink_directory_and_lock_rejected(tmp_path):
    root = tmp_path / "real"
    root.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        PortOwnership([("head", "localhost", 20001)], link)
    target = tmp_path / "target"
    target.write_text("unrelated")
    (root / "20001.lock").symlink_to(target)
    with pytest.raises(OSError):
        PortOwnership([("head", "localhost", 20001)], root)
    assert target.read_text() == "unrelated"


def test_nonprivate_directory_rejected(tmp_path):
    root = tmp_path / "locks"
    root.mkdir(mode=0o755)
    with pytest.raises(RuntimeError, match="private"):
        PortOwnership([("head", "localhost", 20001)], root)


def test_native_start_conflict_precedes_any_service_start(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    import nemo_gym.cli.env as env

    config = {"head_server": {"host": "localhost", "port": 20001}}
    monkeypatch.setattr(env, "get_global_config_dict", lambda **kwargs: config)
    monkeypatch.setattr(env.GlobalConfigDictParser, "raise_on_no_server_instances", lambda *args: None)
    monkeypatch.setattr(env, "_server_spinup_timeout_seconds", lambda cfg: 10)
    monkeypatch.setattr(env, "_configured_servers", lambda cfg: [])
    monkeypatch.setattr(env, "PortOwnership", lambda listeners: PortOwnership(listeners, tmp_path / "locks"))
    startup = MagicMock()
    monkeypatch.setattr(env, "SupervisorStartup", lambda: startup)
    monkeypatch.setattr(
        env, "telemetry_config_from_global_config", lambda cfg: pytest.fail("must reject before startup")
    )
    owner = PortOwnership([("other head", "0.0.0.0", 20001)], tmp_path / "locks")
    try:
        with pytest.raises(RuntimeError, match="owns local port"):
            env.RunHelper().start(None)
        # Failed contender's cleanup must not release or signal the original owner.
        with pytest.raises(RuntimeError, match="owns local port"):
            PortOwnership([("probe", "localhost", 20001)], tmp_path / "locks")
    finally:
        owner.close()


def test_native_start_failure_releases_acquired_ports(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    import nemo_gym.cli.env as env

    config = {"head_server": {"host": "localhost", "port": 20001}}
    monkeypatch.setattr(env, "get_global_config_dict", lambda **kwargs: config)
    monkeypatch.setattr(env.GlobalConfigDictParser, "raise_on_no_server_instances", lambda *args: None)
    monkeypatch.setattr(env, "_server_spinup_timeout_seconds", lambda cfg: 10)
    monkeypatch.setattr(env, "_configured_servers", lambda cfg: [])
    monkeypatch.setattr(env, "PortOwnership", lambda listeners: PortOwnership(listeners, tmp_path / "locks"))
    monkeypatch.setattr(env, "SupervisorStartup", MagicMock)

    def fail(cfg):
        with pytest.raises(RuntimeError, match="owns local port"):
            PortOwnership([("probe", "localhost", 20001)], tmp_path / "locks")
        raise ValueError("injected setup failure")

    monkeypatch.setattr(env, "telemetry_config_from_global_config", fail)
    with pytest.raises(ValueError, match="injected setup failure"):
        env.RunHelper().start(None)
    PortOwnership([("next", "localhost", 20001)], tmp_path / "locks").close()


def test_default_namespace_ignores_different_process_tmpdir(tmp_path):
    code = """import sys
from nemo_gym.cli.port_ownership import PortOwnership
owner=PortOwnership([("head","0.0.0.0",20739)])
print("ready",flush=True)
sys.stdin.readline()
owner.close()
"""
    env = dict(os.environ, TMPDIR=str(tmp_path))
    process = subprocess.Popen(
        [sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        with pytest.raises(RuntimeError, match="owns local port"):
            PortOwnership([("other", "localhost", 20739)])
    finally:
        process.stdin.write("done\n")
        process.stdin.flush()
        process.wait(timeout=10)
    PortOwnership([("released", "localhost", 20739)]).close()
