# SPDX-License-Identifier: Apache-2.0
import json
import os
import sys

import pytest

from responses_api_agents.metasyn_native.execution import run_creator


@pytest.mark.asyncio
async def test_creator_output_retained(tmp_path):
    await run_creator([sys.executable, '-c', 'print("native-output")'], tmp_path)
    assert (tmp_path / 'native-runtime.log').read_text().strip() == 'native-output'
    assert json.loads((tmp_path / 'execution.json').read_text())['return_code'] == 0


@pytest.mark.asyncio
async def test_failure_does_not_become_score(tmp_path):
    with pytest.raises(RuntimeError):
        await run_creator([sys.executable, '-c', 'raise RuntimeError("creator-failure")'], tmp_path)
    assert 'creator-failure' in (tmp_path / 'native-runtime.log').read_text()
    assert json.loads((tmp_path / 'execution.json').read_text())['scientific_score_assigned'] is False


@pytest.mark.asyncio
async def test_timeout_settles_real_child(tmp_path):
    pid_path = tmp_path / 'child.pid'
    code = ('import subprocess,sys,time;from pathlib import Path;'
            'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);'
            f'Path({str(pid_path)!r}).write_text(str(p.pid));time.sleep(60)')
    with pytest.raises(TimeoutError):
        await run_creator([sys.executable, '-c', code], tmp_path, timeout_s=1)
    pid = int(pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert json.loads((tmp_path / 'execution.json').read_text())['error_type'] == 'timeout'


@pytest.mark.asyncio
async def test_success_does_not_leave_redirected_child(tmp_path):
    import asyncio
    pid_path = tmp_path / 'orphan.pid'
    code = ('import subprocess,sys;from pathlib import Path;'
            'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"],'
            'stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);'
            f'Path({str(pid_path)!r}).write_text(str(p.pid))')
    await run_creator([sys.executable, '-c', code], tmp_path)
    pid = int(pid_path.read_text())
    for _ in range(20):
        try: os.kill(pid, 0)
        except ProcessLookupError: break
        await asyncio.sleep(0.05)
    else:
        os.kill(pid, 9)
        raise AssertionError('Creator child survived successful parent cleanup')
