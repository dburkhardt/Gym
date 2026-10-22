# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.scivisagentbench.transport import normalize_tool_message_nulls
from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming


def captured_request():
    return json.loads((Path(__file__).parent / 'test_fixtures/tool_message_request.json').read_text())


def test_actual_captured_request_only_removes_absent_tool_calls():
    original = captured_request()
    preserved = copy.deepcopy(original)
    with pytest.raises(ValidationError):
        NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(original)
    normalized = normalize_tool_message_nulls(original)
    NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(normalized)
    assert original == preserved
    expected = copy.deepcopy(original)
    assert expected['messages'][3]['tool_calls'] is None
    del expected['messages'][3]['tool_calls']
    assert normalized == expected


@pytest.mark.parametrize('value', [[], {}, 'invalid', [{'id': 'real-call'}]])
def test_non_null_tool_calls_remain_rejected(value):
    request = captured_request()
    request['messages'][3]['tool_calls'] = value
    normalized = normalize_tool_message_nulls(request)
    assert normalized == request
    with pytest.raises(ValidationError):
        NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(normalized)


def test_other_fields_and_roles_are_not_normalized():
    request = captured_request()
    request['messages'][2]['tool_calls'] = None
    request['messages'][3]['unexpected'] = None
    normalized = normalize_tool_message_nulls(request)
    assert normalized['messages'][2]['tool_calls'] is None
    assert 'unexpected' in normalized['messages'][3]
    with pytest.raises(ValidationError):
        NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(normalized)


@pytest.mark.asyncio
async def test_native_adapter_submits_normalized_request_once_and_preserves_original(tmp_path):
    from benchmarks.scivisagentbench.transport import MODEL, NativeClient

    original = captured_request()
    calls = []

    async def sender(request):
        NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(request)
        calls.append(request)
        return {'id': 'control', 'model': MODEL, 'choices': [
            {'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'Done'}}]}

    path = tmp_path / 'receipt.jsonl'
    client = NativeClient(sender, path, max_tokens=16384)
    kwargs = copy.deepcopy(original)
    kwargs['stream'] = True
    await client.create(**kwargs)
    assert len(calls) == 1
    dispatch = json.loads(path.read_text().splitlines()[0])
    assert dispatch['creator_request'] == original
    assert dispatch['request'] == normalize_tool_message_nulls(original) == calls[0]
