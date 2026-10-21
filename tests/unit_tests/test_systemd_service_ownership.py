# SPDX-License-Identifier: Apache-2.0
"""Dedicated-scope protocol controls; real systemd acceptance is opt-in Linux."""

import json
import os
from types import SimpleNamespace

import pytest

import nemo_gym.cli.service_ownership as ownership
from tests.unit_tests.test_service_ownership import receipt


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(ownership.sys, "platform", "linux")
    monkeypatch.setattr(ownership, "owned_members", lambda *_: pytest.fail("global process census forbidden"))


def properties(value):
    unit = ownership._unit(value)
    return {
        "Id": unit,
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "InvocationID": "a" * 32,
        "Description": ownership._description(value),
        "ControlGroup": f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{unit}",
    }


def test_explicit_scope_command_preserves_environment_without_secret_argv(tmp_path, linux, monkeypatch):
    monkeypatch.setattr(ownership, "_unit_properties", lambda _: None)
    owner = ownership.ServiceOwnership(tmp_path, {}, mode="systemd")
    env = {"SECRET_FIXTURE": "never-persist-me"}
    command = owner.command("printf '%s' \"$SECRET_FIXTURE\"", env)
    value = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert command[:3] == ["systemd-run", "--user", "--scope"]
    assert "never-persist-me" not in repr(command) + json.dumps(value)
    assert env["SECRET_FIXTURE"] == "never-persist-me"
    assert value["systemd"] is None
    owner.close()


@pytest.mark.parametrize(
    "field,new",
    [
        ("InvocationID", "b" * 32),
        ("Description", "other"),
        ("ControlGroup", "/user.slice/session-8685.scope"),
        ("Id", "other.scope"),
    ],
)
def test_mismatched_scope_refused_before_stop(tmp_path, linux, monkeypatch, field, new):
    path = receipt(tmp_path, mode="systemd", systemd=None)
    value = json.loads(path.read_text())
    props = properties(value)
    value["systemd"] = {
        "unit": props["Id"],
        "invocation_id": props["InvocationID"],
        "control_group": props["ControlGroup"],
    }
    path.write_text(json.dumps(value))
    before = path.read_bytes()
    props[field] = new
    monkeypatch.setattr(ownership, "_unit_properties", lambda _: props)
    monkeypatch.setattr(ownership.subprocess, "run", lambda *_a, **_k: pytest.fail("must not stop"))
    with pytest.raises(RuntimeError, match="scope"):
        ownership.ServiceOwnership(tmp_path, value["binding"], mode="systemd")
    assert path.read_bytes() == before


def test_scope_reconcile_stops_only_bound_unit_and_requires_empty_cgroup(tmp_path, linux, monkeypatch):
    path = receipt(tmp_path, mode="systemd", systemd=None)
    value = json.loads(path.read_text())
    props = properties(value)
    state = {"stopped": False}
    calls = []
    monkeypatch.setattr(ownership, "_unit_properties", lambda _: None if state["stopped"] else props)
    monkeypatch.setattr(ownership, "_scope_populated", lambda _: not state["stopped"])

    def stop(argv, **kwargs):
        calls.append(argv)
        state["stopped"] = True
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ownership.subprocess, "run", stop)
    owner = ownership.ServiceOwnership(tmp_path, value["binding"], mode="systemd")
    owner.close()
    assert calls == [["systemctl", "--user", "stop", ownership._unit(value)]]
    assert json.loads(path.read_text())["cleanup_confirmed"] is True


def test_missing_unit_with_populated_recorded_group_refused(tmp_path, linux, monkeypatch):
    path = receipt(tmp_path, mode="systemd", systemd=None)
    value = json.loads(path.read_text())
    p = properties(value)
    value["systemd"] = {"unit": p["Id"], "invocation_id": p["InvocationID"], "control_group": p["ControlGroup"]}
    path.write_text(json.dumps(value))
    monkeypatch.setattr(ownership, "_unit_properties", lambda _: None)
    monkeypatch.setattr(ownership, "_scope_populated", lambda _: True)
    with pytest.raises(RuntimeError, match="populated"):
        ownership.ServiceOwnership(tmp_path, value["binding"], mode="systemd")


def test_no_implicit_receipt_mode_migration(tmp_path, linux):
    path = receipt(tmp_path)
    with pytest.raises(RuntimeError, match="binding mismatch"):
        ownership.ServiceOwnership(tmp_path, json.loads(path.read_text())["binding"], mode="systemd")


def test_unavailable_user_manager_is_not_missing_unit(monkeypatch):
    monkeypatch.setattr(
        ownership.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=1, stdout="", stderr="bus absent")
    )
    with pytest.raises(RuntimeError, match="Cannot inspect"):
        ownership._unit_properties("nemogym-" + "a" * 32 + ".scope")


def test_stopped_scope_cannot_hide_surviving_root(tmp_path, linux, monkeypatch):
    path = receipt(tmp_path, mode="systemd", systemd=None)
    value = json.loads(path.read_text())
    props = properties(value)
    monkeypatch.setattr(ownership, "_unit_properties", lambda _: props)
    monkeypatch.setattr(ownership, "_scope_populated", lambda _: False)
    monkeypatch.setattr(ownership, "matching", lambda _: object())
    monkeypatch.setattr(ownership.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=0))
    with pytest.raises(RuntimeError, match="outlived"):
        ownership._stop_scope(value)
