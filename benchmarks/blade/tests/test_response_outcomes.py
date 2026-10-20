# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from benchmarks.blade import native_phase


def response(content, finish):
    return {
        "id": "control",
        "object": "chat.completion",
        "created": 0,
        "model": native_phase.MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": content, "reasoning_content": "reasoning"},
            }
        ],
        "usage": {
            "prompt_tokens": 12234,
            "completion_tokens": 7350,
            "total_tokens": 19584,
            "completion_tokens_details": {"reasoning_tokens": 7350},
        },
    }


@pytest.mark.parametrize(
    "content,finish,kind,raises",
    [
        (None, "stop", "model_no_visible_answer", False),
        ("", "stop", "model_no_visible_answer", False),
        (None, "length", "model_output_budget_exhausted", True),
        ("answer", "stop", "visible_answer", False),
    ],
)
def test_captured_response_classification_preserves_raw_and_no_retry(
    tmp_path, monkeypatch, content, finish, kind, raises
):
    body = response(content, finish)
    calls = []

    def send(*args):
        calls.append(args)
        return body

    monkeypatch.setattr(native_phase, "post_json", send)
    directory = tmp_path / "calls"
    transport = native_phase.Transport(
        "http://localhost/ng-rollout/control/v1", "lineage", "digest", directory, socket_path="unused", token="control"
    )
    if raises:
        with pytest.raises(RuntimeError):
            transport.create(model=native_phase.MODEL)
        with pytest.raises(RuntimeError):
            transport.create(model=native_phase.MODEL)
        assert json.loads((directory / "000000.failure.json").read_text())["response_captured"] is True
    else:
        result = transport.create(model=native_phase.MODEL)
        assert result.choices[0].message.content == content
    assert len(calls) == 1
    assert json.loads((directory / "000000.response.json").read_text()) == body
    outcome = json.loads((directory / "000000.outcome.json").read_text())
    assert outcome["kind"] == kind and outcome["creator_score"] is None
    assert outcome["usage"]["completion_tokens"] == 7350
