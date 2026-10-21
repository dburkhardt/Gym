# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.corebench.operational_review import write_overlay


def native():
    return {
        "instance_id": "corebench::capsule-1683542",
        "reward": 0.0,
        "mask_sample": False,
        "metadata": {
            "config": {"agent": {"name": "oracle", "import_path": None}},
            "agent_info": {"name": "oracle"},
            "verifier_result": {"rewards": {"reward": 0.0}},
        },
        "ng_model_call_capture": {"metrics": {"num_calls": 0}, "calls": []},
        "response": {"output": []},
    }


def test_overlay_preserves_raw_and_excludes_quality(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(native()) + "\n")
    before = raw.read_bytes()
    report = tmp_path / "report.json"
    report.write_text("{}\n")
    summary = write_overlay(raw, report, tmp_path / "derived")
    assert raw.read_bytes() == before
    assert summary["counts"]["raw_scored_records"] == 1
    assert summary["counts"]["quality_selected_records"] == 0
    assert summary["per_metric"]["reward"]["reviewed_denominator"] == 0
    assert summary["per_metric"]["reward"]["invalid"] == 1
    revised = json.loads((tmp_path / "derived/reviewed-results.jsonl").read_text())
    assert revised["reward"] == revised["metadata"]["verifier_result"]["rewards"]["reward"] == 0
    with pytest.raises(FileExistsError):
        write_overlay(raw, report, tmp_path / "derived")


@pytest.mark.parametrize("change", ["model", "calls", "unknown_calls", "response", "reward", "instance", "report"])
def test_non_control_refused(tmp_path, change):
    value = copy.deepcopy(native())
    if change == "model":
        value["metadata"]["config"]["agent"]["name"] = "terminus-2"
    elif change == "calls":
        value["ng_model_call_capture"]["calls"] = [{"id": "call"}]
    elif change == "unknown_calls":
        value["ng_model_call_capture"]["metrics"]["num_calls"] = None
    elif change == "response":
        value["response"]["output"] = [{"text": "answer"}]
    elif change == "reward":
        value["reward"] = 1
    elif change == "instance":
        value["instance_id"] = "other::task"
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(value))
    report = tmp_path / "report.json"
    report.write_text('{"answer": 1}' if change == "report" else "{}")
    with pytest.raises(ValueError, match="Operational review refused"):
        write_overlay(raw, report, tmp_path / "derived")
    assert not (tmp_path / "derived/reviewed-results.jsonl").exists()


def test_future_manifest_follows_results_symlink_and_refuses_empty(tmp_path):
    script = Path(__file__).parents[1] / "controls/1683542/solve-v2.sh"
    section = script.read_text().split("find -L /results", 1)[1].split("# No fabricated", 1)[0]
    command = "set -eu\nfind -L /results" + section
    output = tmp_path / "actual-results"
    output.mkdir()
    link = tmp_path / "results"
    link.symlink_to(output, target_is_directory=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = command.replace("/results", str(link)).replace("/workspace", str(workspace))
    assert subprocess.run(["bash", "-c", command]).returncode != 0
    picture = output / "Figure.2.png"
    picture.write_bytes(b"test image bytes")
    assert subprocess.run(["bash", "-c", command]).returncode == 0
    result = (workspace / "scientific-figures.sha256").read_text()
    assert hashlib.sha256(picture.read_bytes()).hexdigest() in result
    assert str(link / picture.name) in result
