# SPDX-License-Identifier: Apache-2.0
import json
import os
from types import SimpleNamespace

import pytest

import nemo_gym.cli.service_ownership as ownership
from tests.unit_tests.test_service_ownership import receipt


def systemd_receipt(directory):
    path = receipt(directory, mode="systemd", cleanup_confirmed=True)
    value = json.loads(path.read_text())
    unit = ownership._unit(value)
    value["systemd"] = {
        "unit": unit,
        "invocation_id": "a" * 32,
        "control_group": f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{unit}",
    }
    path.write_text(json.dumps(value))
    return path


@pytest.fixture(autouse=True)
def no_mutations(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("inspection must not mutate or enumerate unrelated processes")

    monkeypatch.setattr(ownership.ServiceOwnership, "__init__", forbidden)
    monkeypatch.setattr(ownership, "owned_members", forbidden)
    monkeypatch.setattr(ownership, "_stop_scope", forbidden)
    monkeypatch.setattr(ownership.psutil.Process, "send_signal", forbidden)


def test_clean_scope_readonly(tmp_path, monkeypatch):
    path = systemd_receipt(tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(ownership, "_unit_properties", lambda _: None)
    result = ownership.inspect_service_ownership(tmp_path, {"source": "frozen"})
    assert result["status"] == "clean"
    assert result["external_containers_checked"] is False
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("change", [{"host": {}}, {"binding": {}}, {"version": True}, {"token": "bad"}])
def test_malformed_identity_unknown(tmp_path, change):
    receipt(tmp_path, **change)
    assert ownership.inspect_service_ownership(tmp_path, {"source": "frozen"})["status"] == "unknown"


@pytest.mark.parametrize("error", [PermissionError("manager inaccessible"), RuntimeError("invocation changed")])
def test_unavailable_scope_unknown(tmp_path, monkeypatch, error):
    receipt(tmp_path, mode="systemd", cleanup_confirmed=True)

    def fail(_):
        raise error

    monkeypatch.setattr(ownership, "_unit_properties", fail)
    result = ownership.inspect_service_ownership(tmp_path, {"source": "frozen"})
    assert result["status"] == "unknown"
    assert result["records"][0]["reason"] == str(error)


def test_live_root_outside_missing_unit_not_clean(tmp_path, monkeypatch):
    receipt(tmp_path, mode="systemd", cleanup_confirmed=True)
    monkeypatch.setattr(ownership, "matching", lambda value: object())
    monkeypatch.setattr(ownership, "_unit_properties", lambda _: None)
    assert ownership.inspect_service_ownership(tmp_path, {"source": "frozen"})["status"] != "clean"


def test_process_absence_remains_unknown(tmp_path):
    receipt(tmp_path, cleanup_confirmed=True)
    assert ownership.inspect_service_ownership(tmp_path, {"source": "frozen"})["status"] == "unknown"


def test_missing_directory_not_created(tmp_path):
    directory = tmp_path / "missing"
    assert ownership.inspect_service_ownership(directory, {})["status"] == "unknown"
    assert not directory.exists()


def test_native_shutdown_receipt_and_original_error(tmp_path, monkeypatch):
    from nemo_gym.cli.eval import _shutdown_native_services

    directory = tmp_path / "native_services"
    directory.mkdir()
    systemd_receipt(directory)
    monkeypatch.setattr(ownership, "_unit_properties", lambda _: None)
    owner = SimpleNamespace(directory=directory, binding={"source": "frozen"})

    def fail():
        raise RuntimeError("shutdown error")

    helper = SimpleNamespace(_service_ownership=owner, shutdown=fail)
    with pytest.raises(ValueError, match="execution error"):
        try:
            raise ValueError("execution error")
        finally:
            _shutdown_native_services(helper)
    assert json.loads((tmp_path / "native-service-cleanup.json").read_text())["status"] == "clean"
    with pytest.raises(RuntimeError, match="shutdown error"):
        _shutdown_native_services(helper)


def test_incomplete_launch_not_clean(tmp_path, monkeypatch):
    receipt(tmp_path, mode="systemd", systemd=None, root=None, cleanup_confirmed=True)
    monkeypatch.setattr(ownership, "_unit_properties", lambda _: None)
    assert ownership.inspect_service_ownership(tmp_path, {"source": "frozen"})["status"] == "unknown"


def test_inspection_publication_failure_preserves_primary(tmp_path, monkeypatch):
    import nemo_gym.sandbox.lifecycle as lifecycle
    from nemo_gym.cli.eval import _shutdown_native_services

    def unavailable(*args):
        raise PermissionError("receipt destination unavailable")

    monkeypatch.setattr(lifecycle, "_publish", unavailable)
    helper = SimpleNamespace(_service_ownership=SimpleNamespace(directory=tmp_path, binding={}), shutdown=lambda: None)
    with pytest.raises(ValueError, match="original"):
        try:
            raise ValueError("original")
        finally:
            _shutdown_native_services(helper)
