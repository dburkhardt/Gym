# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from transport import MODEL, NativeClient, stream_chunk

BASE = {'id': 'control', 'model': MODEL, 'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'Done'}}], 'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}

class TransportTests(unittest.TestCase):
    def test_text_and_usage_preserved(self):
        c = stream_chunk(BASE)
        self.assertEqual(c['choices'][0]['delta']['content'], 'Done')
        self.assertEqual(c['usage'], BASE['usage'])

    def test_tool_arguments_preserved(self):
        r = copy.deepcopy(BASE)
        call = {'id': 'call1', 'type': 'function', 'function': {'name': 'open_file', 'arguments': '{"file_path":"data/a.tif"}'}}
        r['choices'][0] = {'finish_reason': 'tool_calls', 'message': {'content': None, 'tool_calls': [call]}}
        self.assertEqual(stream_chunk(r)['choices'][0]['delta']['tool_calls'], [{**call, 'index': 0}])

    def test_invalid_response_saved_before_rejection_without_retry(self):
        async def check():
            for modification in ['wrong_model', 'length', 'empty']:
                with tempfile.TemporaryDirectory() as d:
                    r = copy.deepcopy(BASE)
                    if modification == 'wrong_model': r['model'] = 'other'
                    if modification == 'length': r['choices'][0]['finish_reason'] = 'length'
                    if modification == 'empty': r['choices'][0]['message']['content'] = ''
                    calls = []
                    async def send(request): calls.append(request); return r
                    path = Path(d)/'receipt.jsonl'
                    client = NativeClient(send, path, max_tokens=16384)
                    with self.assertRaises(ValueError):
                        await client.create(model=MODEL, messages=[], stream=True)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(json.loads(path.read_text().splitlines()[-1])['response'], r)
        asyncio.run(check())

if __name__ == '__main__': unittest.main()
