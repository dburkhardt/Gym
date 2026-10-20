# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invoke the pinned creator agent; only model transport and paths are adapted."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from benchmarks.scivisagentbench.transport import MODEL, NativeClient


async def run_native(source: Path, config: dict, task: dict, output: Path, sender, max_tokens: int) -> dict:
    if config.get('model') != MODEL or config.get('provider') != 'openai':
        raise ValueError('exact Super OpenAI-compatible configuration required')
    if not config.get('servers'):
        raise ValueError('native MCP servers must be preflighted')
    output.mkdir(parents=True, exist_ok=False)
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
    agent = module.NapariMCPAgent(config)
    try:
        await agent.setup()
        await agent.prepare_task(case.get_task_config(agent.agent_mode))
        prompt = case.get_task_description(agent.agent_mode)
        # Only question/path metadata is sent to the solver. source_case.assert stays host-side.
        (output/'solver-prompt.txt').write_text(prompt)
        result = await agent.run_task(prompt, case.get_task_config(agent.agent_mode))
        record = result.to_dict()
        (output/'native-solver-result.json').write_text(json.dumps(record, indent=2))
        return record
    finally:
        await agent.cleanup_task(case.get_task_config(agent.agent_mode))
        await agent.teardown()
        mcp_client.AsyncInferenceClient = original
