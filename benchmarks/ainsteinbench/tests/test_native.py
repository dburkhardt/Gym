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


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [TimeoutError("native timeout"), 137])
async def test_native_scorer_failure_never_becomes_zero(tmp_path, failure):
    import hashlib

    from benchmarks.ainsteinbench.harbor_native import AInsteinVerifier

    task_id = "MSB_pyscf_pyscf_pr2373"

    class Env:
        async def download_dir(self, _, target):
            target.mkdir()
            (target / "answer.patch").write_text("")
            (target / "outcome.json").write_text(
                json.dumps(
                    {"task_id": task_id, "status": "submitted", "patch_sha256": hashlib.sha256(b"").hexdigest()}
                )
            )

        async def upload_file(self, *args):
            pass

        async def exec(self, *args, **kwargs):
            if isinstance(failure, Exception):
                raise failure
            return SimpleNamespace(return_code=failure, stdout="", stderr="killed")

    verifier = AInsteinVerifier(
        source=SOURCE,
        task=SimpleNamespace(
            paths=SimpleNamespace(task_dir=Path(task_id)),
            config=SimpleNamespace(verifier=SimpleNamespace(environment_mode="separate")),
        ),
        trial_paths=SimpleNamespace(verifier_dir=tmp_path),
        environment=Env(),
    )
    with pytest.raises((TimeoutError, RuntimeError)):
        await verifier.verify()
    assert not (tmp_path / "native/score.json").exists()
    if isinstance(failure, int):
        assert json.loads((tmp_path / "native/execution.json").read_text())["return_code"] == 137


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["printf '%s' hello", 'echo "it\'s quoted"', "printf 'a b\\n'"])
async def test_tool_command_matches_docker_sdk_shlex(command):
    import shlex

    from benchmarks.ainsteinbench.harbor_native import execute_tool

    calls = []

    class Env:
        async def exec(self, actual, **kwargs):
            calls.append(shlex.split(actual))
            return SimpleNamespace(return_code=0, stdout="ok", stderr="")

    # Docker SDK splits string commands with shlex.split before exec_create.
    try:
        expected = shlex.split(f"bash -c '{command}'")
    except ValueError as error:
        assert await execute_tool(Env(), "exec_command", {"command": command}) == f"Error: {error}"
        assert calls == []
    else:
        await execute_tool(Env(), "exec_command", {"command": command})
        assert calls == [expected]


def test_native_trajectory_all_calls_and_usage(tmp_path):
    from harbor.models.agent.context import AgentContext

    from benchmarks.ainsteinbench.trajectory import publish
    from responses_api_agents.harbor_agent.utils import HarborAgentUtils

    directory = tmp_path / "native"
    directory.mkdir()
    (directory / "identity.json").write_text(json.dumps({"lineage": "durable"}))
    for i in range(2):
        (directory / f"call-{i}.response.json").write_text(
            json.dumps(
                {
                    "usage": {
                        "prompt_tokens": 10 + i,
                        "completion_tokens": 3,
                        "prompt_tokens_details": {"cached_tokens": 2},
                    }
                }
            )
        )
    messages = [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "start"},
        {
            "role": "assistant",
            "content": "thinking",
            "tool_calls": [{"id": "tool1", "function": {"name": "read_file", "arguments": '{"path":"a"}'}}],
        },
        {"role": "tool", "tool_call_id": "tool1", "content": "result"},
        {"role": "assistant", "content": "answer"},
    ]
    context = AgentContext()
    trajectory = publish(directory, messages, context)
    assert context.n_input_tokens == 21 and context.n_output_tokens == 6 and context.n_cache_tokens == 4
    assert context.metadata["native_model_calls"] == 2
    assert [s["extra"]["logical_call_id"] for s in trajectory["steps"] if s["source"] == "agent"] == [
        "ainstein:durable:0",
        "ainstein:durable:1",
    ]
    outputs = HarborAgentUtils.trajectory_to_responses(trajectory)
    assert sum(o["type"] == "message" for o in outputs) == 2
    assert any(o["type"] == "function_call_output" and o["output"] == "result" for o in outputs)
    assert HarborAgentUtils.extract_usage({}, trajectory)["total_tokens"] == 27
    (directory / "call-1.response.json").write_text("{}")
    unknown = AgentContext()
    new = publish(directory, messages, unknown)
    assert unknown.n_input_tokens is None and "final_metrics" not in new
    assert not unknown.metadata["provider_usage_complete"]


@pytest.mark.asyncio
async def test_native_scorer_accepts_absent_stderr(tmp_path):
    import hashlib

    from benchmarks.ainsteinbench.harbor_native import AInsteinVerifier

    task_id = "MSB_pyscf_pyscf_pr2373"

    class Env:
        async def download_dir(self, _, target):
            target.mkdir()
            (target / "answer.patch").write_text("patch")
            (target / "outcome.json").write_text(
                json.dumps(
                    {"task_id": task_id, "status": "submitted", "patch_sha256": hashlib.sha256(b"patch").hexdigest()}
                )
            )

        async def upload_file(self, *args):
            pass

        async def exec(self, *args, **kwargs):
            return SimpleNamespace(return_code=0, stdout="PASSED a\n", stderr=None)

    verifier = AInsteinVerifier(
        source=SOURCE,
        task=SimpleNamespace(
            paths=SimpleNamespace(task_dir=Path(task_id)),
            config=SimpleNamespace(verifier=SimpleNamespace(environment_mode="separate")),
        ),
        trial_paths=SimpleNamespace(verifier_dir=tmp_path),
        environment=Env(),
    )
    result = await verifier.verify()
    assert result.rewards == {"score": 100, "tests_passed": 1, "tests_total": 1}
