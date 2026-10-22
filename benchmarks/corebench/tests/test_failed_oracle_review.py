# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from benchmarks.corebench.failed_oracle_review import review, write_overlay
from benchmarks.corebench.tests.test_operational_review import native


def inputs(tmp_path, change=None):
    r = native()
    if change == "agent":
        r["metadata"]["config"]["agent"]["name"] = "terminus-2"
    if change == "calls":
        r["ng_model_call_capture"]["calls"] = [{"id": "x"}]
    if change == "unknown":
        r["ng_model_call_capture"]["metrics"]["num_calls"] = None
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(r) + "\n")
    code = tmp_path / "exit-code.txt"
    code.write_text("0" if change == "success" else "1")
    log = tmp_path / "scientific-workflow.log"
    log.write_text("other" if change == "log" else "+ Rscript 01_data_processing.R\nExecution halted\n")
    return raw, code, log


def test_excludes_failed_oracle(tmp_path):
    raw, code, log = inputs(tmp_path)
    original = raw.read_bytes()
    summary = write_overlay(raw, code, log, tmp_path / "derived")
    assert summary["counts"]["quality_selected_records"] == 0
    assert summary["per_metric"]["reward"]["reviewed_denominator"] == 0
    assert raw.read_bytes() == original
    assert not (tmp_path / "derived/empty-report.json").exists()
    r = json.loads((tmp_path / "derived/reviewed-results.jsonl").read_text())
    assert r["reward"] == 0 and r["mask_sample"]


@pytest.mark.parametrize("change", ["agent", "calls", "unknown", "success", "log"])
def test_refuses_noncontrol(tmp_path, change):
    with pytest.raises(ValueError, match="Operational review refused"):
        write_overlay(*inputs(tmp_path, change), tmp_path / "derived")
    assert not (tmp_path / "derived/reviewed-results.jsonl").exists()


@pytest.mark.parametrize("name", ["exit-code.txt", "scientific-workflow.log", "raw-result.jsonl"])
def test_hashes_bind_evidence(tmp_path, name):
    raw, code, log = inputs(tmp_path)
    out = tmp_path / "derived"
    write_overlay(raw, code, log, out)
    (out / name).write_bytes((out / name).read_bytes() + b" ")
    r = json.loads(raw.read_text())
    with pytest.raises(ValueError):
        review(trial_dir=out, instance_id=r["instance_id"], trial_result=r["metadata"])
