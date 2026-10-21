# SPDX-License-Identifier: Apache-2.0
import sys

import pytest

from resources_servers.bioplanner import app


@pytest.fixture(autouse=True)
def reset_limit(monkeypatch):
    monkeypatch.setattr(app.LocalProvider, "_semaphores", {})


@pytest.mark.asyncio
async def test_payload_and_metrics_preserved_without_evaluating_answer(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json,sys\np=json.load(sys.stdin)\nprint(json.dumps({'metrics':{'argument_recall':.5},'answer':p['answer']}))"
    )
    payload = {"answer": "__import__('sys').exit(99)"}
    result = await app.run_score(payload, python=sys.executable, worker=worker)
    assert result == {"metrics": {"argument_recall": 0.5}, **payload}


@pytest.mark.asyncio
async def test_timeout_kills_scientific_descendant(tmp_path):
    worker = tmp_path / "worker.py"
    pidfile = tmp_path / "child.pid"
    worker.write_text(
        "import subprocess,sys,time,pathlib\np=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid))\ntime.sleep(60)\n"
    )
    result = await app.run_score({}, python=sys.executable, worker=worker, timeout_s=0.5)
    assert result["metrics"] is None and result["scorer_error_type"] == "timeout"
    pid = int(pidfile.read_text())
    # A killed child can briefly remain a zombie until its parent is reaped.
    import subprocess

    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    assert not state or state.startswith("Z")


@pytest.mark.asyncio
async def test_failure_and_malformed_output_do_not_become_scores(tmp_path):
    worker = tmp_path / "worker.py"
    for code in ("raise RuntimeError('broken verifier')", "print('not json')", "print('[]')"):
        worker.write_text(code)
        result = await app.run_score({}, python=sys.executable, worker=worker)
        assert result["metrics"] is None and result["scorer_error"]
