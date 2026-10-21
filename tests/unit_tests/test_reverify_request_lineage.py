# SPDX-License-Identifier: Apache-2.0
import copy
import json
from pathlib import Path

import pytest

from nemo_gym.config_types import ConfigError
from nemo_gym.rollout_reverification import InputRolloutPair, _build_verify_payload, _retain_generation_failure


@pytest.mark.parametrize("nested", [False, True])
def test_saved_generation_preserved_with_current_reference(nested):
    supplied = {"input": [{"role": "user", "content": "question"}]}
    executed = {
        **supplied,
        "temperature": 0.1,
        "max_output_tokens": 16384,
        "tools": [{"type": "function", "name": "x"}],
    }
    fields = {"responses_create_params": supplied, "verifier_metadata": {"answer": "new"}}
    inputs = {"task_input": {"task_data": fields}} if nested else fields
    rollout = {
        "responses_create_params": executed,
        "response": {"output": "saved"},
        "reward": 1,
        "verifier_metadata": {"answer": "old"},
    }
    before = copy.deepcopy((inputs, rollout))
    payload = _build_verify_payload(InputRolloutPair(input=inputs, rollout=rollout))
    assert payload["responses_create_params"] == executed
    assert payload["verifier_metadata"] == {"answer": "new"}
    assert payload["response"] == rollout["response"]
    proof = payload["_ng_reverify_request_provenance"]
    assert proof["changed_top_level_fields"] == ["max_output_tokens", "temperature", "tools"]
    assert proof["executed_sha256"] != proof["supplied_sha256"]
    assert "question" not in json.dumps(proof)
    result = _retain_generation_failure(payload, {"reward": 0, "responses_create_params": supplied})
    assert result["responses_create_params"] == executed
    assert result["reverification_request_provenance"] == proof
    assert result["reward"] == 0
    assert (inputs, rollout) == before


@pytest.mark.parametrize("bad", [None, [], "invalid", 3, {"temperature": float("nan")}])
def test_malformed_saved_request_fails_closed(bad):
    with pytest.raises(ConfigError):
        _build_verify_payload(InputRolloutPair(input={}, rollout={"response": {}, "responses_create_params": bad}))


def test_missing_saved_request_keeps_existing_fallback():
    pair = InputRolloutPair(input={"responses_create_params": {"input": "question"}}, rollout={"response": {}})
    assert _build_verify_payload(pair) == {**pair.input, "response": {}}


def test_equal_request_defaults_have_equal_hashes():
    request = {"input": "q", "temperature": None}
    payload = _build_verify_payload(
        InputRolloutPair(
            input={"responses_create_params": request}, rollout={"response": {}, "responses_create_params": request}
        )
    )
    proof = payload["_ng_reverify_request_provenance"]
    assert proof["changed_top_level_fields"] == []
    assert proof["executed_sha256"] == proof["supplied_sha256"]


def test_actual_preserved_chemiq_request():
    root = Path(__file__).resolve().parents[4]
    source = root / "outputs/broad-eval-campaign/chemiq/pilot-5-v2.jsonl"
    inputs = root / "outputs/broad-eval-campaign/scientific-agents/chemiq/chemiq-5.jsonl"
    if not source.exists():
        pytest.skip("Optional historical campaign artifacts unavailable")
    row = json.loads(source.read_text().splitlines()[0])
    candidates = [json.loads(line) for line in inputs.read_text().splitlines()]
    task = next(x for x in candidates if x["uuid"] == row["uuid"])
    payload = _build_verify_payload(InputRolloutPair(input=task, rollout=row))
    assert payload["responses_create_params"] == row["responses_create_params"]
    assert payload["responses_create_params"]["temperature"] == 0.1
    assert payload["responses_create_params"]["max_output_tokens"] == 16384
    assert payload["verifier_metadata"] == task["verifier_metadata"]
    assert payload["response"] == row["response"]
