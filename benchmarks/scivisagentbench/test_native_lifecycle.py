# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from benchmarks.scivisagentbench.native import cleanup_native, validate_source

class LifecycleTests(unittest.TestCase):
    def test_cleanup_failure_still_tears_down_and_restores_factory(self):
        calls=[]
        class Agent:
            async def cleanup_task(self, config): calls.append('cleanup'); raise ValueError('cleanup broke')
            async def teardown(self): calls.append('teardown'); raise ValueError('teardown broke')
        original=object(); module=SimpleNamespace(AsyncInferenceClient=object())
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                asyncio.run(cleanup_native(Agent(),{},module,original,Path(d)))
            self.assertIs(module.AsyncInferenceClient,original)
            self.assertEqual(calls,['cleanup','teardown'])
            self.assertEqual(len(json.loads((Path(d)/'lifecycle.json').read_text())['cleanup_errors']),2)

    def test_primary_error_preserved_with_secondary_cleanup_error(self):
        class Agent:
            async def cleanup_task(self, config): raise ValueError('cleanup')
            async def teardown(self): pass
        original=object(); module=SimpleNamespace(AsyncInferenceClient=object())
        with tempfile.TemporaryDirectory() as d:
            asyncio.run(cleanup_native(Agent(),{},module,original,Path(d),ValueError('solver')))
            self.assertIs(module.AsyncInferenceClient,original)
            self.assertIn('solver',json.loads((Path(d)/'lifecycle.json').read_text())['primary_error'])

    def test_source_revision_rejected_before_import(self):
        with patch('benchmarks.scivisagentbench.native.subprocess.check_output',return_value='wrong'):
            with self.assertRaisesRegex(ValueError,'revision_mismatch'):
                validate_source(Path('/unused'))

if __name__=='__main__':unittest.main()
