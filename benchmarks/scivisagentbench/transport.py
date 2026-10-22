# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport-only compatibility for the pinned TinyAgent streaming consumer.

The provider is called once, nonstreaming. Its unmodified request/response is
persisted before validation; the complete reply is emitted as one native chunk.
Tool calls, IDs, arguments, final text and reported usage are not regenerated.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace


MODEL = 'nvidia/nvidia/nemotron-3.5-super-vl'


def normalize_tool_message_nulls(request: dict) -> dict:
    """Omit TinyAgent's absent assistant-only field on tool response messages.

    Do not discard non-null or unknown fields: the strict model schema must
    still reject them. Preserve content, call IDs, names, and all other values.
    """
    messages = request.get('messages')
    if not isinstance(messages, list):
        return request
    return {**request, 'messages': [
        {key: value for key, value in message.items() if key != 'tool_calls'}
        if isinstance(message, dict) and message.get('role') == 'tool'
        and 'tool_calls' in message and message['tool_calls'] is None else message
        for message in messages
    ]}


def validate_response(response: dict) -> None:
    if response.get('model') != MODEL:
        raise ValueError('provider_model_mismatch')
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError('invalid_choices')
    choice = choices[0]
    if choice.get('finish_reason') not in ('stop', 'tool_calls'):
        raise ValueError('incomplete_provider_response')
    message = choice.get('message') or {}
    if not message.get('content') and not message.get('tool_calls'):
        raise ValueError('empty_provider_response')
    if message.get('tool_calls') and choice.get('finish_reason') != 'tool_calls':
        raise ValueError('inconsistent_tool_finish')


def stream_chunk(response: dict) -> dict:
    validate_response(response)
    message = response['choices'][0]['message']
    calls = [{**call, 'index': i} for i, call in enumerate(message.get('tool_calls') or [])]
    return {
        'id': response['id'], 'created': response.get('created', 0), 'model': response['model'],
        'object': 'chat.completion.chunk', 'usage': response.get('usage'),
        'choices': [{'index': 0, 'finish_reason': response['choices'][0]['finish_reason'],
                     'delta': {'role': 'assistant', 'content': message.get('content'),
                               'tool_calls': calls or None}}],
    }


class NativeClient:
    """HF client-shaped adapter; injected sender must perform exactly one request."""
    def __init__(self, sender, receipt_path: Path, *, max_tokens: int):
        self.sender = sender
        self.receipt_path = Path(receipt_path)
        self.max_tokens = max_tokens
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def create(self, **kwargs):
        if kwargs.get('model') != MODEL:
            raise ValueError('requested_model_mismatch')
        if kwargs.pop('stream', None) is not True:
            raise ValueError('native_consumer_expected_stream')
        kwargs.pop('stream_options', None)
        kwargs.update(stream=False, max_tokens=self.max_tokens)
        # HF message dataclasses inherit dict; JSON serialization is checked before dispatch.
        creator_request = json.loads(json.dumps(kwargs))
        request = normalize_tool_message_nulls(creator_request)
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with self.receipt_path.open('a') as f:
            f.write(json.dumps({'phase': 'dispatch', 'time': time.time(), 'request': request,
                                'creator_request': creator_request}) + '\n')
        try:
            response = await self.sender(request)
        except Exception as exc:
            with self.receipt_path.open('a') as f:
                f.write(json.dumps({'phase': 'error', 'error': repr(exc)}) + '\n')
            raise
        with self.receipt_path.open('a') as f:
            f.write(json.dumps({'phase': 'response', 'response': response}) + '\n')
        chunk = stream_chunk(response)
        async def iterator():
            from huggingface_hub.inference._generated.types import ChatCompletionStreamOutput
            yield ChatCompletionStreamOutput.parse_obj_as_instance(chunk)
        return iterator()
