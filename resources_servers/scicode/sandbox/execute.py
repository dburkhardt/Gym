# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted bounded wrapper inside an owned sandbox; generated children cannot reach host secrets."""

import json
import os
import signal
import subprocess
import sys
import tempfile


with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
    child = subprocess.Popen(
        [sys.executable, "/input/program.py"], stdout=stdout, stderr=stderr, start_new_session=True
    )
    timed_out = False
    try:
        child.wait(timeout=float(sys.argv[1]))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
    stderr.seek(0, 2)
    stderr.seek(max(0, stderr.tell() - 2000))
    tail = stderr.read().decode("utf-8", errors="replace")
    print(
        json.dumps(
            {
                "passed": child.returncode == 0 and not timed_out,
                "error": "timeout" if timed_out else tail,
                "timed_out": timed_out,
                "return_code": child.returncode,
            }
        )
    )
