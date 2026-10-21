# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys
from pathlib import Path

from responses_api_agents.bioagent_native.app import selected_interpreter


def test_selected_venv_is_actual_execution_environment():
    executable, identity = selected_interpreter(Path(sys.executable))
    assert str(executable) == os.path.abspath(sys.executable)
    result = subprocess.run(
        [str(executable), "-c", "import sys;print(sys.prefix)"], check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == sys.prefix
    assert identity["path"] == str(executable)
    assert identity["binary_path"] == str(executable.resolve())
