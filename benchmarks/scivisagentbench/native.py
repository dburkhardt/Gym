# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invoke the pinned creator agent; only model transport and paths are adapted."""
from __future__ import annotations

import importlib.util
import json
import sys
import subprocess
import hashlib
from pathlib import Path

from benchmarks.scivisagentbench.transport import MODEL, NativeClient


SOURCE_REVISION = 'b9435d13e9ff979226b1754e6c467efeb026978a'


def validate_source(source: Path) -> dict:
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != SOURCE_REVISION:
        raise ValueError('creator_source_revision_mismatch')
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', 'HEAD', '--', 'benchmark', 'src'], check=True, capture_output=True)
    files = ['benchmark/tiny_agent/agent.py', 'benchmark/tiny_agent/mcp_client.py',
             'benchmark/evaluation_framework/agents/napari_mcp_agent.py',
             'benchmark/evaluation_framework/unified_runner.py']
    return {'revision': revision, 'sha256': {p: hashlib.sha256((source/p).read_bytes()).hexdigest() for p in files}}


async def cleanup_native(agent, task_config, mcp_client, original, output: Path, primary_error=None):
    errors = []
    try:
        if agent is not None:
            try:
                await agent.cleanup_task(task_config)
            except BaseException as exc:
                errors.append({'phase':'cleanup_task','error':repr(exc)})
            finally:
                try:
                    await agent.teardown()
                except BaseException as exc:
                    errors.append({'phase':'teardown','error':repr(exc)})
    finally:
        mcp_client.AsyncInferenceClient = original
        (output/'lifecycle.json').write_text(json.dumps({'primary_error':repr(primary_error) if primary_error else None,'cleanup_errors':errors,'client_factory_restored':True},indent=2))
    if errors and primary_error is None:
        raise RuntimeError('native cleanup failed; see lifecycle.json')


async def run_native(source: Path, config: dict, task: dict, output: Path, sender, max_tokens: int) -> dict:
    provenance = validate_source(source)
    if config.get('model') != MODEL or config.get('provider') != 'openai':
        raise ValueError('exact Super OpenAI-compatible configuration required')
    if not config.get('servers'):
        raise ValueError('native MCP servers must be preflighted')
    output.mkdir(parents=True, exist_ok=False)
    (output/'source-provenance.json').write_text(json.dumps(provenance,indent=2))
    sys.path.insert(0, str(source/'benchmark'))
    sys.path.insert(0, str(source/'benchmark/tiny_agent'))
    from evaluation_framework.unified_runner import YAMLTestCase
    import mcp_client
    # Load only the selected agent, avoiding imports for unrelated optional harnesses.
    spec = importlib.util.spec_from_file_location('scivis_native_napari_adapter', source/'benchmark/evaluation_framework/agents/napari_mcp_agent.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = YAMLTestCase(task['source_case'], task['case_name'], task['solver_workspace'],
                        task['solver_workspace'], 'eval_visualization_tasks')
    original = mcp_client.AsyncInferenceClient
    clients = []
    def factory(**kwargs):
        client = NativeClient(sender, output/'solver-provider.jsonl', max_tokens=max_tokens)
        clients.append(client)
        return client
    mcp_client.AsyncInferenceClient = factory
    agent = None
    primary_error = None
    task_config = {}
    try:
        agent = module.NapariMCPAgent(config)
        task_config = case.get_task_config(agent.agent_mode)
        await agent.setup()
        await agent.prepare_task(case.get_task_config(agent.agent_mode))
        prompt = case.get_task_description(agent.agent_mode)
        # Only question/path metadata is sent to the solver. source_case.assert stays host-side.
        (output/'solver-prompt.txt').write_text(prompt)
        result = await agent.run_task(prompt, case.get_task_config(agent.agent_mode))
        # Use the creator's own result serialization for efficiency/token components.
        # Its host-only case directory is separate from the solver's writable workspace.
        if task.get('judge_case_dir'):
            agent.save_test_result(result, task['judge_case_dir'], task['case_name'], prompt)
        record = result.to_dict()
        (output/'native-solver-result.json').write_text(json.dumps(record, indent=2))
        return record
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        await cleanup_native(agent, task_config, mcp_client, original, output, primary_error)
