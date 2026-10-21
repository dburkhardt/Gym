import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from nemo_gym.cli import env
from nemo_gym.cli.setup_command import validate_server_launcher
from nemo_gym.config_types import ConfigError
from nemo_gym.global_config import SKIP_VENV_IF_PRESENT_KEY_NAME, UV_VENV_DIR_KEY_NAME


def layout(tmp_path):
    component = tmp_path / "source" / "responses_api_agents" / "agent"
    component.mkdir(parents=True)
    config = {UV_VENV_DIR_KEY_NAME: str(tmp_path / "venvs"), SKIP_VENV_IF_PRESENT_KEY_NAME: True}
    venv = tmp_path / "venvs" / "responses_api_agents" / "agent" / ".venv"
    return component, config, venv


def test_missing_uv_rejected_before_startup_side_effects(tmp_path, monkeypatch):
    component, config, _ = layout(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(env, "get_global_config_dict", lambda **kw: config)
    monkeypatch.setattr(env.GlobalConfigDictParser, "raise_on_no_server_instances", lambda *a: None)
    monkeypatch.setattr(env, "_configured_servers", lambda c: [("agent", "", "", {}, Path("app.py"), component)])
    effects = [Mock(side_effect=AssertionError("must not start")) for _ in range(4)]
    monkeypatch.setattr(env, "initialize_ray", effects[0])
    monkeypatch.setattr(env, "PortOwnership", effects[1])
    monkeypatch.setattr(env, "start_h2_ping_sidecar", effects[2])
    monkeypatch.setattr(env.HeadServer, "run_webserver", effects[3])
    with pytest.raises(ConfigError, match="uv is not executable"):
        env.RunHelper.__new__(env.RunHelper)._start(None)
    assert all(not item.called for item in effects)


def test_existing_qualified_interpreter_does_not_require_uv(tmp_path, monkeypatch):
    component, config, venv = layout(tmp_path)
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(sys.executable)
    (venv / "bin/activate").write_text("")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    report = validate_server_launcher(component, config, "agent")
    assert report == {"server": "agent", "python": str(venv / "bin/python"), "reuse_existing_venv": True, "uv": None}


def test_setup_needed_uses_actual_launcher_path(tmp_path, monkeypatch):
    component, config, venv = layout(tmp_path)
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    report = validate_server_launcher(component, config, "agent")
    assert report["uv"] == str(uv)
    assert report["reuse_existing_venv"] is False
    assert not venv.exists()


@pytest.mark.parametrize("kind", ["nonexecutable", "directory", "missing", "broken_symlink"])
def test_unusable_interpreter_fails_without_setup_executable(tmp_path, monkeypatch, kind):
    component, config, venv = layout(tmp_path)
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/activate").write_text("")
    python = venv / "bin/python"
    if kind == "directory":
        python.mkdir()
    elif kind == "nonexecutable":
        python.write_text("not executable")
    elif kind == "broken_symlink":
        python.symlink_to(tmp_path / "absent")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(ConfigError, match="non-executable|uv is not executable"):
        validate_server_launcher(component, config, "agent")


def test_activation_directory_rejected(tmp_path, monkeypatch):
    component, config, venv = layout(tmp_path)
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(sys.executable)
    (venv / "bin/activate").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(ConfigError, match="unreadable venv activation"):
        validate_server_launcher(component, config, "agent")
