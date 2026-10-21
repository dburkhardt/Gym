import importlib.util
import hashlib
import os
import json
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from benchmarks.chemiq.observation import check


@pytest.mark.parametrize(
    "case,invalid,reward,calls",
    [
        ("correct", False, True, 1),
        ("wrong", False, False, 1),
        ("empty", False, False, 0),
        ("404", False, False, 1),
        ("400", False, False, 1),
        ("timeout", True, False, 1),
        ("429", True, False, 1),
        ("503", True, False, 1),
        ("connection", True, False, 1),
        ("json", True, False, 1),
        ("alarm", True, False, 1),
        ("cache", False, False, 0),
        ("local_alarm", True, False, 0),
    ],
)
def test_actual_creator(tmp_path, monkeypatch, case, invalid, reward, calls):
    repo = Path(os.environ.get("CHEMIQ_CREATOR_REPO", Path(__file__).resolve().parents[4] / "chemiq-upstream"))
    source = repo / "utils/answer_verifier.py"
    if not source.exists():
        pytest.skip("Set CHEMIQ_CREATOR_REPO to pinned 63826f6 creator checkout")
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "4e0df4c0881396b7c3831627875e0f492200c09a0613af17d0fb97f59704cbfa"
    )
    spec = importlib.util.spec_from_file_location("observed_creator", source)
    creator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(creator)
    monkeypatch.chdir(tmp_path)
    task = {"uuid": "control", "verifier_metadata": {"answer": "CCO"}}
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps({"uuid": "control", "answer": "CCO", "verification_method": "opsin", "answer_range": ""}) + "\n"
    )
    verifier = creator.AnswerVerifier(questions)
    if case == "local_alarm":
        verifier.question_dict["control"]["verification_method"] = "exact_match"
        verifier._check_exact_match = lambda *args: time.sleep(3)
    if case == "cache":
        verifier._opsin_cache["ethanol"] = False
    count = []

    def get(*args, **kwargs):
        count.append((args, kwargs))
        if case == "timeout":
            raise requests.Timeout("fixture")
        if case == "connection":
            raise requests.ConnectionError("fixture")
        if case == "alarm":
            time.sleep(3)  # The unchanged creator's real two-second SIGALRM interrupts this.
        response = requests.Response()
        response.status_code = int(case) if case.isdigit() else 200
        response._content = (
            b"not json" if case == "json" else json.dumps({"smiles": "CC" if case == "wrong" else "CCO"}).encode()
        )
        return response

    creator.requests = SimpleNamespace(get=get, exceptions=requests.exceptions)
    task["verifier_metadata"] = dict(verifier.question_dict["control"])
    original_requests, original_signal = creator.requests, creator.signal
    original_handler = signal.getsignal(signal.SIGALRM)
    result, evidence = check(
        creator,
        verifier,
        task,
        "" if case == "empty" else "ethanol",
        source_pin="63826f64990802e87f1b11b9192606a20838d522",
    )
    assert result["is_correct"] is reward
    assert len(count) == calls
    assert any(e["dependency_failure"] or e.get("verifier_failure", False) for e in evidence["events"]) is invalid
    assert creator.requests is original_requests and creator.signal is original_signal
    assert signal.getsignal(signal.SIGALRM) == original_handler
    if calls:
        assert verifier._opsin_cache["ethanol"] == ("CCO" if case == "correct" else "CC" if case == "wrong" else False)
    if case == "alarm":
        assert any(e["kind"] == "creator_alarm" for e in evidence["events"])

    # Compare the same actual creator without observation, including cache effects.
    observed_cache = dict(verifier._opsin_cache)
    verifier._opsin_cache = {"ethanol": False} if case == "cache" else {}
    baseline = verifier.check_answer(task["uuid"], "" if case == "empty" else "ethanol")
    assert baseline == result
    assert verifier._opsin_cache == observed_cache
    assert len(count) == calls * 2

    if case == "local_alarm":
        assert evidence["events"] == [
            {"kind": "creator_alarm", "phase": "local_verifier", "dependency_failure": False, "verifier_failure": True}
        ]

    tampered = dict(task, verifier_metadata={**task["verifier_metadata"], "answer": "changed"})
    before_calls = len(count)
    with pytest.raises(ValueError, match="reference differs"):
        check(creator, verifier, tampered, "ethanol", source_pin="63826f64990802e87f1b11b9192606a20838d522")
    assert len(count) == before_calls
