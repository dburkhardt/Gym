# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.corebench.templates import item_diagnostics as diagnostic


@pytest.fixture(scope="module")
def scorer():
    root = Path(os.environ.get("COREBENCH_ASSET_ROOT", "../../outputs/broad-eval-campaign"))
    path = (
        root
        / "foundation/corebench-native/3137115-live-proposal/preflight-evidence/core313-live-e628-v2/prepared/tasks/capsule-3137115/tests/evaluations.py"
    )
    if not path.is_file():
        pytest.skip("Pinned creator scorer assets not staged")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == diagnostic.SCORER_SHA256
    spec = importlib.util.spec_from_file_location("core_original_item_diagnostics_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(scorer, refs, prediction):
    baseline = copy.deepcopy(prediction)
    expected = scorer.eval_result_json(copy.deepcopy(refs), baseline)
    actual = copy.deepcopy(prediction)
    result, report = diagnostic.evaluate(scorer.eval_result_json, copy.deepcopy(refs), actual)
    assert result == expected
    assert actual == baseline
    assert report["status"] == "available", report
    assert report["original_components"] == result
    assert (
        sum((x["observation"] or {}).get("original_counter_increment", 0) for x in report["items"])
        == result["correct_written_answers"] + result["correct_vision_answers"]
    )
    return report


def test_real_rounding_interval(scorer):
    report = compare(scorer, [{"fig value": 0.62}] * 3, {"fig value": 0.6169})
    item = report["items"][0]
    assert item["numeric_interval"] == [0.62, 0.62]
    assert item["prediction_normalized"] == 0.6169
    assert item["original_verdict"] is False
    assert item["status"] == "evaluated"


def test_variable_references_and_percent_conversion(scorer):
    report = compare(scorer, [{"x": x} for x in [8, 10, 12]], {"x": "10%"})
    item = report["items"][0]
    assert item["prediction_raw"] == "10%" and item["prediction_normalized"] == 10
    assert item["numeric_interval"][0] < 10 < item["numeric_interval"][1]
    assert item["original_verdict"] is True


def test_strings_lists_missing_extra(scorer):
    report = compare(
        scorer,
        [{"text": "Yes", "list": [1, 2], "missing": 3}] * 3,
        {"text": "YES", "list": [1, 2], "extra": "anything"},
    )
    assert [x["original_verdict"] for x in report["items"]] == [True, True, None]
    assert report["items"][2]["status"] == "not_reported"
    assert report["unrecognized_report_keys"] == ["extra"]


def test_early_caught_exception_is_not_later_failed_verdict(scorer):
    report = compare(scorer, [{"x": 1, "later": 2}] * 3, {"x": {}, "later": 2})
    assert report["items"][0]["status"] == "comparison_exception"
    assert report["items"][1]["status"] == "not_reached"
    assert all(x["original_verdict"] is None for x in report["items"])


def test_nonfinite_interval_is_explicit(scorer):
    report = compare(scorer, [{"x": 1}], {"x": 1})
    assert report["items"][0]["interval_nonfinite"] is True
    assert report["items"][0]["original_verdict"] is False
    json.dumps(report, allow_nan=False)


def test_original_exception_preserved(scorer):
    with pytest.raises(IndexError):
        diagnostic.evaluate(scorer.eval_result_json, [], {})
    assert sys.gettrace() is None


def test_preexisting_trace_untouched_single_call(scorer):
    calls = []

    def tracing(frame, event, arg):
        if frame.f_code is scorer.eval_result_json.__code__ and event == "call":
            calls.append(event)
        return tracing

    previous = sys.gettrace()
    try:
        sys.settrace(tracing)
        result, report = diagnostic.evaluate(scorer.eval_result_json, [{"x": 1}] * 3, {"x": 1})
        assert sys.gettrace() is tracing
    finally:
        sys.settrace(previous)
    assert calls == ["call"] and result["correct_written_answers"] == 1
    assert report == {"status": "unavailable", "reason": "preexisting_trace"}


@pytest.mark.parametrize("method", ["finish", "report"])
def test_observer_failure_does_not_change_score(scorer, monkeypatch, method):
    def fail(*args):
        raise RuntimeError("diagnostic failure")

    monkeypatch.setattr(diagnostic.Observation, method, fail)
    result, report = diagnostic.evaluate(scorer.eval_result_json, [{"x": 1}] * 3, {"x": "1"})
    assert result["correct_written_answers"] == 1 and report["status"] == "unavailable"
    assert sys.gettrace() is None


def test_unknown_source_does_not_change_single_call(tmp_path):
    source = tmp_path / "different.py"
    source.write_text(
        'calls=0\ndef eval_result_json(refs, reported):\n global calls\n calls+=1\n reported["mutated"]=True\n return {"answer":7}\n'
    )
    namespace = {}
    exec(compile(source.read_bytes(), str(source), "exec"), namespace)
    prediction = {}
    result, report = diagnostic.evaluate(namespace["eval_result_json"], [], prediction)
    assert result == {"answer": 7} and namespace["calls"] == 1 and prediction == {"mutated": True}
    assert report["status"] == "unavailable"


def test_ast_gate_unavailable_without_score_change(scorer, monkeypatch):
    monkeypatch.setattr(diagnostic.ast, "unparse", lambda node: "different")
    result, report = diagnostic.evaluate(scorer.eval_result_json, [{"x": 1}] * 3, {"x": 1})
    assert result["correct_written_answers"] == 1 and report["status"] == "unavailable"


def test_publication_failure_is_diagnostic_only(scorer, tmp_path, capsys):
    result, report = diagnostic.evaluate(scorer.eval_result_json, [{"x": 1}] * 3, {"x": 1})
    diagnostic.publish(
        report,
        tmp_path,
        scorer_path=scorer.__file__,
        reference_path=tmp_path / "absent",
        report_path=tmp_path / "absent",
    )
    assert result["correct_written_answers"] == 1
    assert "diagnostics unavailable" in capsys.readouterr().err


def test_actual_check_script_writes_evaluator_only_artifact(scorer, tmp_path):
    import shutil

    template = Path(diagnostic.__file__).parent
    tests = tmp_path / "tests"
    tests.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    logs = tmp_path / "logs/verifier"
    shutil.copyfile(scorer.__file__, tests / "evaluations.py")
    shutil.copyfile(template / "item_diagnostics.py", tests / "item_diagnostics.py")
    (tests / "answers.json").write_text(json.dumps([{"fig value": 0.62}] * 3))
    (workspace / "report.json").write_text(json.dumps({"fig value": 0.6169}))
    source = (
        (template / "check.py")
        .read_text()
        .replace("/tests", str(tests))
        .replace("/workspace", str(workspace))
        .replace("/logs/verifier", str(logs))
    )
    script = tmp_path / "check.py"
    script.write_text(source)
    child = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    assert json.loads((logs / "reward.json").read_text()) == {"reward": 0.0}
    report = json.loads((logs / "item_diagnostics.json").read_text())
    assert report["status"] == "available" and report["items"][0]["original_verdict"] is False
    assert report["sha256"]["scorer"] == diagnostic.SCORER_SHA256
    assert not list(workspace.glob("*diagnostic*")) and not (workspace / "answers.json").exists()


def test_active_observer_calls_original_once(scorer, monkeypatch):
    calls = []
    original = scorer.np.mean

    def mean(values):
        calls.append(list(values))
        return original(values)

    monkeypatch.setattr(scorer.np, "mean", mean)
    result, report = diagnostic.evaluate(scorer.eval_result_json, [{"x": 1}] * 3, {"x": 1})
    assert calls == [[1, 1, 1]]
    assert result["correct_written_answers"] == 1 and report["status"] == "available"


def test_loaded_code_mismatch_unavailable(scorer, monkeypatch):
    monkeypatch.setattr(
        scorer.eval_result_json, "__code__", scorer.eval_result_json.__code__.replace(co_name="changed")
    )
    result, report = diagnostic.evaluate(scorer.eval_result_json, [{"x": 1}] * 3, {"x": 1})
    assert result["correct_written_answers"] == 1 and report["status"] == "unavailable"
