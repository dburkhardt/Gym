# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import httpx
import pytest

from benchmarks.biabench.process import MODEL, guarded_evaluator, process_identity


class Evaluator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.client = SimpleNamespace(close=lambda: None)


def module():
    return SimpleNamespace(LLMEvaluator=Evaluator, _load_local_env_files=lambda: None)


def test_native_budget_and_model_are_not_overridden(tmp_path):
    creator = module()
    with guarded_evaluator(
        creator,
        socket_path="/unused",
        token="control",
        lineage="attempt",
        digest="digest",
        output=tmp_path / "calls",
        transport=httpx.MockTransport(lambda r: None),
    ):
        with pytest.raises(ValueError, match="budget"):
            creator.LLMEvaluator(model=MODEL, max_tokens=16384)
        with pytest.raises(ValueError, match="model"):
            creator.LLMEvaluator(model="other", max_tokens=2048)
        evaluator = creator.LLMEvaluator(model=MODEL, max_tokens=2048, temperature=0.1)
        assert evaluator.kwargs["max_tokens"] == 2048
        assert evaluator.kwargs["temperature"] == 0.1
        assert evaluator.kwargs["max_retries"] == 1
        assert evaluator.client.max_retries == 0
    assert creator.LLMEvaluator is Evaluator


def test_ambiguous_request_cannot_be_resent(tmp_path):
    creator = module()
    seen = []

    def disconnect(request):
        seen.append(json.loads(request.content))
        raise httpx.ReadError("accepted then disconnected")

    with guarded_evaluator(
        creator,
        socket_path="/unused",
        token="control",
        lineage="attempt",
        digest="digest",
        output=tmp_path / "calls",
        transport=httpx.MockTransport(disconnect),
    ):
        evaluator = creator.LLMEvaluator(model=MODEL, max_tokens=2048)
        for _ in range(2):
            with pytest.raises(Exception):
                evaluator.client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": "question"}]
                )
    assert len(seen) == 1
    assert len(list((tmp_path / "calls").glob("*.request.json"))) == 1


def test_identity_distinguishes_fresh_attempt_and_saved_evidence(tmp_path):
    (tmp_path / "answer").write_text("original")
    _, first = process_identity(tmp_path, "attempt1")
    assert first == process_identity(tmp_path, "attempt1")[1]
    assert first != process_identity(tmp_path, "attempt2")[1]
    (tmp_path / "answer").write_text("changed")
    assert first != process_identity(tmp_path, "attempt1")[1]
