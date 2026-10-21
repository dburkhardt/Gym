# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.ainsteinbench.harbor_native import NativeIterationLimitError, creator_loop
from benchmarks.ainsteinbench.prepare import stage
from benchmarks.ainsteinbench.source import checked_source, creator_methods, parse_log


SOURCE = Path(__file__).resolve().parents[4] / "ainsteinbench-pinned-source"


def test_all_frozen_tasks_preserved_and_et_blocked(tmp_path):
    rows, _ = checked_source(SOURCE)
    output = stage(SOURCE, tmp_path)
    staged = [json.loads(x) for x in output.read_text().splitlines()]
    assert [r["task_id"] for r in staged] == list(rows)
    assert len(staged) == 4
    assert sum(bool(r["verifier_metadata"]["blocked_reason"]) for r in staged) == 2
    import tomllib

    from harbor.models.task.config import TaskConfig

    for row in staged:
        task = tmp_path / "tasks" / row["task_id"]
        config = TaskConfig.model_validate(tomllib.loads((task / "task.toml").read_text()))
        assert config.environment.network_mode.value == "no-network"
        assert config.verifier.environment_mode == "separate"
        assert not (task / "environment").exists()


@pytest.mark.asyncio
async def test_creator_loop_matches_original_trace_and_truncation():
    raw = next(iter(checked_source(SOURCE)[0].values()))
    raw = copy.deepcopy(raw)
    raw["environment"]["working_directory"] = "/home/pyscf"
    replies = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path":"x","offset":9}'},
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {"message": {"role": "assistant", "content": "```diff\n--- a/x\n+++ b/x\n```", "tool_calls": None}}
            ]
        },
    ]
    captured = []

    async def completion(ordinal, messages, tools):
        captured.append(copy.deepcopy(messages))
        return replies[ordinal]

    class Env:
        async def exec(self, *args, **kwargs):
            return SimpleNamespace(return_code=0, stdout="x" * 12000, stderr="")

    answer = await creator_loop(raw=raw, source=SOURCE, environment=Env(), completion=completion, save=lambda x: None)
    assert answer == "--- a/x\n+++ b/x"
    assert captured[1][-1]["content"] == "x" * 10000
    # Execute the original creator solve_question AST with controlled environment
    # and scripted model objects; compare every request's messages exactly.
    tree = ast.parse((SOURCE / "scripts/run_agent.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DockerAgent")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "solve_question")
    ns = {
        "Question": object,
        "json": json,
        "logging": __import__("logging"),
        "TOOLS": creator_methods(SOURCE)["TOOLS"],
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "creator", "exec"), ns)
    actual = []

    def create(**kwargs):
        actual.append(copy.deepcopy(kwargs["messages"]))
        message = replies[len(actual) - 1]["choices"][0]["message"]
        tools = message["tool_calls"]
        msg = SimpleNamespace(
            content=message["content"],
            tool_calls=[SimpleNamespace(id=t["id"], function=SimpleNamespace(**t["function"])) for t in tools]
            if tools
            else None,
            to_dict=lambda: copy.deepcopy(message),
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    methods = creator_methods(SOURCE)
    obj = SimpleNamespace(
        initialize_container=lambda q: None,
        cleanup_container=lambda: None,
        build_initial_prompt=lambda q: methods["_build_msb_prompt"](None, q),
        model="test",
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        execute_tool=lambda *a: "x" * 12000,
        _extract_code_from_markdown=lambda s: methods["_extract_code_from_markdown"](None, s),
    )
    assert ns["solve_question"](obj, SimpleNamespace(**raw)) == answer
    assert actual == captured


@pytest.mark.asyncio
async def test_budget_exhaustion_and_transport_failure_distinct():
    raw = next(iter(checked_source(SOURCE)[0].values()))
    calls = []

    async def completion(ordinal, *args):
        calls.append(ordinal)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": str(ordinal), "function": {"name": "unknown", "arguments": "{}"}}],
                    }
                }
            ]
        }

    with pytest.raises(NativeIterationLimitError):
        await creator_loop(raw=raw, source=SOURCE, environment=None, completion=completion, save=lambda x: None)
    assert calls == list(range(30))

    async def failed(*args):
        raise ConnectionError("ambiguous")

    with pytest.raises(ConnectionError):
        await creator_loop(raw=raw, source=SOURCE, environment=None, completion=failed, save=lambda x: None)


def test_exact_creator_parser_retains_unique_test_ids():
    result = parse_log(SOURCE, "PASSED a\nPASSED a\nFAILED b - Assertion\nPASSED c\n")
    assert result == {
        "passed_count": 2,
        "failed_count": 1,
        "skipped_count": 0,
        "passed_tests": ["a", "c"],
        "failed_tests": ["b"],
        "skipped_tests": [],
    }


@pytest.mark.asyncio
async def test_early_empty_response_is_not_budget_exhaustion():
    from benchmarks.ainsteinbench.harbor_native import NativeNoSolutionError

    raw = next(iter(checked_source(SOURCE)[0].values()))

    async def empty(*args):
        return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": None}}]}

    with pytest.raises(NativeNoSolutionError):
        await creator_loop(raw=raw, source=SOURCE, environment=None, completion=empty, save=lambda x: None)
