# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nemo_gym import PARENT_DIR
from nemo_gym.cli.env import _server_launch_command
from nemo_gym.global_config import (
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
    UV_VENV_DIR_KEY_NAME,
)
from nemo_gym.telemetry.startup import STARTUP_SETUP_DONE_NS_ENV
from tests.unit_tests.test_global_config import TestGlobalConfig as _TestGlobalConfig


class TestServerLaunchCommand:
    """`gym env start` runs each server with its venv's own interpreter, not the `python` on PATH after
    `source bin/activate`, so a relocated venv whose `bin/activate` names its original prefix cannot win."""

    def _setup_server_dir(self, tmp_path: Path) -> Path:
        # A space in the path catches any unquoted venv or server path in the generated command.
        server_dir = tmp_path / "my checkout" / "resources_servers" / "my_server"
        server_dir.mkdir(parents=True)
        (server_dir / "requirements.txt").write_text("pytest\n")
        (tmp_path / "my checkout" / "pyproject.toml").write_text("")
        return server_dir.absolute()

    @pytest.mark.parametrize("uv_venv_dir", [None, "shared venvs"])
    def test_runs_the_entrypoint_with_the_venv_interpreter(self, tmp_path: Path, uv_venv_dir: str | None) -> None:
        server_dir = self._setup_server_dir(tmp_path)
        global_config_dict = _TestGlobalConfig._default_global_config_dict_values.fget(None) | {
            UV_VENV_DIR_KEY_NAME: str(tmp_path / uv_venv_dir) if uv_venv_dir else str(PARENT_DIR)
        }
        expected_venv = (
            tmp_path / uv_venv_dir / "resources_servers" / "my_server" / ".venv"
            if uv_venv_dir
            else server_dir / ".venv"
        )

        command = _server_launch_command(server_dir, global_config_dict, "my_server", Path("app.py"))

        *setup_lines, launch_line = command.splitlines()
        assert shlex.split(launch_line) == ["&&", "exec", str(expected_venv / "bin" / "python"), "app.py"]
        # The interpreter comes from the same venv that the setup step builds and activates.
        assert f"source {shlex.quote(str(expected_venv / 'bin' / 'activate'))}" in setup_lines[0]
        assert NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME not in command
        assert NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME not in command

    def test_stamps_when_venv_setup_finished_before_launching_the_server(self, tmp_path: Path) -> None:
        """The server cannot see the venv setup that ran before it existed, so the shell stamps its end."""
        server_dir = self._setup_server_dir(tmp_path)
        global_config_dict = _TestGlobalConfig._default_global_config_dict_values.fget(None) | {
            UV_VENV_DIR_KEY_NAME: str(PARENT_DIR)
        }
        venv_python = str(server_dir / ".venv" / "bin" / "python")

        command = _server_launch_command(server_dir, global_config_dict, "my_server", Path("app.py"))

        *_, stamp_line, launch_line = command.splitlines()
        assert f"export {STARTUP_SETUP_DONE_NS_ENV}=$({shlex.quote(venv_python)} -c " in stamp_line
        assert shlex.split(launch_line)[:3] == ["&&", "exec", venv_python]

    def test_the_stamp_is_a_wall_clock_nanosecond_integer_visible_to_the_server(self) -> None:
        """Runs the stamp fragment for real, with the interpreter standing in for the venv's."""
        stamp_fragment = (
            f"export {STARTUP_SETUP_DONE_NS_ENV}=$({shlex.quote(sys.executable)} -c "
            f"'import time; print(time.time_ns())') && {shlex.quote(sys.executable)} -c "
            f"'import os; print(os.environ[\"{STARTUP_SETUP_DONE_NS_ENV}\"])'"
        )
        before_ns = time.time_ns()

        result = subprocess.run(["bash", "-c", stamp_fragment], capture_output=True, text=True, check=True)

        assert before_ns <= int(result.stdout) <= time.time_ns()


def test_server_launch_exec_preserves_owned_process_identity(tmp_path, monkeypatch):
    import nemo_gym.cli.env as env

    monkeypatch.setattr(env, "setup_env_command", lambda *args: "true")
    monkeypatch.setattr(env, "get_venv_path", lambda *args: Path(sys.executable).parent.parent)
    pid_file = tmp_path / "pid.txt"
    entry = tmp_path / "app.py"
    entry.write_text(
        f"import os,time\nfrom pathlib import Path\nPath({str(pid_file)!r}).write_text(str(os.getpid()))\ntime.sleep(30)\n"
    )
    command = env._server_launch_command(tmp_path, {}, "fixture", entry)
    process = subprocess.Popen(command, shell=True, executable="/bin/bash")
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert int(pid_file.read_text()) == process.pid
        process.terminate()
        assert process.wait(timeout=5) != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_shutdown_does_not_signal_reused_exited_root(monkeypatch):
    from unittest.mock import MagicMock

    import nemo_gym.cli.env as env

    stale = MagicMock()
    stale._gym_owned_server_group = True
    stale.poll.return_value = 0
    active = MagicMock()
    runner = env.RunHelper()
    runner._processes = {"exited": stale, "other": active}
    runner._head_server = None
    monkeypatch.setattr(env.os, "killpg", lambda *args: pytest.fail("stale root must not signal a group"))
    runner.shutdown()
    stale._gym_owned_root_identity.is_running.assert_not_called()
    stale.send_signal.assert_not_called()
    active.send_signal.assert_called_once()
