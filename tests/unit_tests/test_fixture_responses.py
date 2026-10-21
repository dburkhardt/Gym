# SPDX-License-Identifier: Apache-2.0
import json

import pytest
from pydantic import ValidationError

from tests.e2e.coordinator.fixture_responses import cancellation_action_fixture, validated_chat_fixture


def test_fixture_validates_actual_model_and_preserves_fields():
    raw = cancellation_action_fixture("nvidia/nvidia/nemotron-3.5-super-vl")
    payload = json.loads(raw)
    assert validated_chat_fixture(payload) == raw
    assert payload["created"] == 0


def test_missing_created_refused_before_services():
    payload = json.loads(cancellation_action_fixture("fixture"))
    del payload["created"]
    with pytest.raises(ValidationError, match="created"):
        validated_chat_fixture(payload)


def test_actual_pinned_terminus_parser_accepts_control_action():
    from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser

    payload = json.loads(cancellation_action_fixture("fixture"))
    parsed = TerminusJSONPlainParser().parse_response(payload["choices"][0]["message"]["content"])
    assert parsed.error == ""
    assert not parsed.is_task_complete
    assert len(parsed.commands) == 1
    assert parsed.commands[0].keystrokes == "printf preserved-by-real-shell > /workspace/preservation.txt\n"
