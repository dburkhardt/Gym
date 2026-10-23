# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.ainsteinbench.harbor_native import NativeIterationLimitError, creator_loop
from benchmarks.ainsteinbench.prepare import stage
from benchmarks.ainsteinbench.source import checked_source, creator_methods, parse_log


SOURCE = Path(os.environ.get("AINSTEIN_SOURCE", Path(__file__).resolve().parents[4] / "ainsteinbench-pinned-source"))
pytestmark = pytest.mark.skipif(not SOURCE.is_dir(), reason="Pinned public creator source required")


def test_all_frozen_tasks_preserved_with_explicit_readiness(tmp_path):
    rows, _ = checked_source(SOURCE)
    output = stage(SOURCE, tmp_path)
    staged = [json.loads(x) for x in output.read_text().splitlines()]
    assert [r["task_id"] for r in staged] == list(rows)
    assert len(staged) == 4
    assert sum(bool(r["verifier_metadata"]["blocked_reason"]) for r in staged) == 1
    import tomllib

    from harbor.models.task.config import TaskConfig

    for row in staged:
        task = tmp_path / "tasks" / row["task_id"]
        config = TaskConfig.model_validate(tomllib.loads((task / "task.toml").read_text()))
        assert config.environment.network_mode.value == "no-network"
        assert config.verifier.environment_mode == "separate"
        assert config.artifacts == ["/submission"]
        assert not (task / "environment").exists()
        if row["verifier_metadata"]["blocked_reason"]:
            assert (
                config.environment.docker_image
                == "sha256:c4884e5e26ca6ba2f0123d9869e4b36ca07a2bcbe7c56ee2b7b9c46fc698ab52"
            )
            assert config.environment.workdir == "/opt/Cactus"


def test_prepared_rows_route_through_native_harbor_contract(tmp_path):
    from responses_api_agents.harbor_agent.app import HarborAgent, HarborAgentConfig, HarborRunRequest

    staged = [json.loads(line) for line in stage(SOURCE, tmp_path).read_text().splitlines()]
    config = HarborAgentConfig(
        name="harbor_agent",
        host="127.0.0.1",
        port=1,
        entrypoint="app.py",
        concurrency=1,
        model_server={"type": "responses_api_models", "name": "policy_model"},
        harbor_datasets={"ainsteinbench": {"local_dataset_path": str(tmp_path / "tasks"), "workdir": "/home/pyscf"}},
        harbor_agent_name=None,
        harbor_agent_import_path="benchmarks.ainsteinbench.harbor_native:AInsteinAgent",
        harbor_agent_kwargs={"source": str(SOURCE)},
        harbor_verifier_import_path="benchmarks.ainsteinbench.harbor_native:AInsteinVerifier",
        harbor_verifier_kwargs={"source": str(SOURCE)},
    )
    server = HarborAgent.model_construct(config=config)
    for row in staged:
        body = HarborRunRequest.model_validate(row)
        alias, task = server._parse_instance_id(body.instance_id)
        job = server._build_job_config(
            alias, task, "nvidia/nvidia/nemotron-3.5-super-vl", "http://127.0.0.1:1/v1", "test", tmp_path / "jobs"
        )
        assert alias == "ainsteinbench"
        assert task == row["task_id"] == row["uuid"] == row["verifier_metadata"]["source_task_id"]
        assert job["datasets"][0]["task_names"] == [task]
        assert Path(job["datasets"][0]["path"]) == tmp_path / "tasks"
        assert (tmp_path / "tasks" / task / "instruction.md").read_bytes() == (task + "\n").encode()
        assert job["retry"]["max_retries"] == 0
        legacy = dict(row, instance_id=task)
        with pytest.raises(ValueError, match="dataset_alias"):
            server._parse_instance_id(HarborRunRequest.model_validate(legacy).instance_id)


def test_new_namespace_preparation_does_not_overwrite_legacy_artifact(tmp_path):
    old = tmp_path / "old"
    path = stage(SOURCE, old)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    legacy = "".join(json.dumps(dict(row, instance_id=row["task_id"])) + "\n" for row in rows).encode()
    path.write_bytes(legacy)
    with pytest.raises(ValueError, match="fresh versioned"):
        stage(SOURCE, old)
    assert path.read_bytes() == legacy
    fresh = stage(SOURCE, tmp_path / "fresh")
    assert fresh.read_bytes() != legacy
    for old_file in (old / "tasks").rglob("*"):
        if old_file.is_file() and not old_file.name.endswith(".stage.lock"):
            assert old_file.read_bytes() == (fresh.parent / "tasks" / old_file.relative_to(old / "tasks")).read_bytes()


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
    if isinstance(failure, Exception):
        assert not (tmp_path / "native/score.json").exists()
    else:
        raw = json.loads((tmp_path / "native/score.json").read_text())
        assert raw["overall_score"] == 0 and not raw["qualification"]["valid"]
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
    (directory / "call-2.intent.json").write_text("{}")
    pending = publish(directory, messages, context)
    assert not context.metadata["provider_usage_complete"]
    assert context.metadata["unresolved_call_ordinals"] == [2]
    assert context.n_input_tokens is None
    assert pending["final_metrics"]["extra"]["provider_usage_complete"] is False
    (directory / "call-2.intent.json").unlink()
    (directory / "call-1.response.json").write_text("{}")
    unknown = AgentContext()
    new = publish(directory, messages, unknown)
    assert unknown.n_input_tokens is None and "total_prompt_tokens" not in new["final_metrics"]
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
            return SimpleNamespace(return_code=0, stdout="PASSED a\nPASSED b\nPASSED c\nPASSED d\n", stderr=None)

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
    assert result.rewards == {"score": 100, "tests_passed": 4, "tests_total": 4}


@pytest.mark.asyncio
async def test_tool_error_observation_preserved_but_cancellation_propagates():
    import asyncio

    from benchmarks.ainsteinbench.harbor_native import execute_tool

    class Env:
        async def exec(self, *args):
            raise RuntimeError("missing path")

    assert await execute_tool(Env(), "read_file", {"path": "/missing"}) == "Error: missing path"

    class Cancelled:
        async def exec(self, *args):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await execute_tool(Cancelled(), "read_file", {"path": "x"})


@pytest.mark.parametrize(
    "parsed,text,code,valid,kind",
    [
        (
            {"passed_count": 0, "failed_count": 0},
            'error: No valid patches in input (allow with "--allow-empty")\nError: git apply failed\n',
            1,
            True,
            "model_patch_rejected",
        ),
        (
            {"passed_count": 0, "failed_count": 0},
            "ModuleNotFoundError: missing_dependency\n",
            1,
            False,
            "unclassified_no_tests",
        ),
        (
            {"passed_count": 0, "failed_count": 0},
            "ERROR collecting tests\n",
            2,
            False,
            "unrecognized_test_execution_status",
        ),
        ({"passed_count": 1, "failed_count": 0}, "PASSED a\n", 0, False, "incomplete_or_failed_test_collection"),
        ({"passed_count": 2, "failed_count": 2}, "PASSED a\nFAILED b\n", 1, True, "creator_tests_scored"),
    ],
)
def test_zero_and_partial_test_qualification(parsed, text, code, valid, kind):
    from benchmarks.ainsteinbench.scoring import qualification

    result = qualification("MSB_pyscf_pyscf_pr2373", parsed, text, code)
    assert result["valid"] is valid and result["kind"] == kind
    assert result["expected_control_test_count"] == 4
    assert result["observed_test_count"] == parsed["passed_count"] + parsed["failed_count"]


@pytest.mark.parametrize("code", [2, 3, 4, 5, 125])
def test_complete_looking_counts_do_not_hide_invalid_exit(code):
    from benchmarks.ainsteinbench.scoring import qualification

    result = qualification("MSB_pyscf_pyscf_pr2373", {"passed_count": 4, "failed_count": 0}, "PASSED a\n", code)
    assert result["valid"] is False
    assert result["kind"] == "unrecognized_test_execution_status"
    assert result["observed_test_count"] == 4


def test_native_config_requires_submission_capture():
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(Path(__file__).parents[1] / "config.yaml")
    assert list(cfg.harbor_agent.responses_api_agents.harbor_agent.harbor_required_artifact_sources) == ["/submission"]


@pytest.mark.parametrize("guard", [True, False, None, "true"])
def test_agent_init_in_process_with_real_ray_worker_arguments(tmp_path, guard):
    import subprocess
    import sys

    script = r"""
import json, sys
from pathlib import Path
from benchmarks.ainsteinbench.harbor_native import AInsteinAgent
flag = json.loads(sys.argv[1])
try:
    agent = AInsteinAgent(source=sys.argv[2], logs_dir=Path(sys.argv[3]),
        api_base="http://127.0.0.1:1/ng-rollout/0-0/v1",
        model_name="nvidia/nvidia/nemotron-3.5-super-vl", gym_provider_dispatch_guard=flag)
except ValueError as error:
    assert flag is not True and "guarded native" in str(error)
    print("guard-refused")
else:
    assert flag is True
    print("guard-accepted")
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            json.dumps(guard),
            str(SOURCE),
            str(tmp_path),
            "--node-ip-address=10.160.4.241",
            "--node-manager-port=43317",
            "--object-store-name=/tmp/ray/control/sockets/plasma_store",
            "--raylet-name=/tmp/ray/control/sockets/raylet",
            "--redis-address=None",
            "--gcs-address=10.160.4.241:46141",
            "--worker-id=control",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert ("guard-accepted" if guard is True else "guard-refused") in result.stdout
    assert "unrecognized arguments" not in result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "annotations",
    [
        None,
        [
            {
                "type": "url_citation",
                "url_citation": {
                    "start_index": 0,
                    "end_index": 1,
                    "title": "fixture",
                    "url": "https://example.invalid/fixture",
                },
            }
        ],
    ],
)
async def test_response_history_roundtrip_through_shared_chat_dispatch(annotations):
    from functools import partial
    from unittest.mock import AsyncMock

    import httpx
    from fastapi import FastAPI

    from benchmarks.ainsteinbench.harbor_native import completion_payload
    from benchmarks.ainsteinbench.native import MODEL
    from nemo_gym.base_responses_api_model import SimpleResponsesAPIModel
    from nemo_gym.openai_utils import NeMoGymChatCompletion

    raw = next(iter(checked_source(SOURCE)[0].values()))
    calls = []
    returned = []
    saved = []

    async def provider(request, params):
        calls.append(params.model_dump(exclude_unset=True))
        message = (
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "fixture reasoning",
                "tool_calls": [
                    {
                        "id": "fixture-tool",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"x"}'},
                    }
                ],
            }
            if len(calls) == 1
            else {"role": "assistant", "content": "```diff\n--- a/x\n+++ b/x\n```"}
        )
        message["annotations"] = annotations
        return NeMoGymChatCompletion.model_validate(
            {
                "id": f"fixture-{len(calls)}",
                "object": "chat.completion",
                "created": 0,
                "model": MODEL,
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            }
        )

    # Exercise the maintained shared dispatcher/parser/response serializer with a
    # deterministic backend. Neither the creator loop nor parser is replaced.
    server = SimpleNamespace(_invoke_chat_completions=provider, _finalize_served_response=AsyncMock())
    api = FastAPI()
    api.post("/v1/chat/completions")(partial(SimpleResponsesAPIModel.chat_completions_dispatch, server))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url="http://fixture") as client:

        async def completion(ordinal, messages, tools):
            payload = completion_payload(messages, tools)
            response = await client.post("/v1/chat/completions", json=payload)
            assert response.status_code == 200, response.text
            body = response.json()
            returned.append(copy.deepcopy(body))
            return body

        class Env:
            async def exec(self, *args, **kwargs):
                return SimpleNamespace(return_code=0, stdout="file content", stderr="")

        answer = await creator_loop(
            raw=raw,
            source=SOURCE,
            environment=Env(),
            completion=completion,
            save=lambda messages: saved.append(copy.deepcopy(messages)),
        )
    assert answer == "--- a/x\n+++ b/x"
    assert len(calls) == 2
    original = returned[0]["choices"][0]["message"]
    assert "annotations" in original and original["annotations"] == annotations
    assert saved[0][-1] == original  # Full creator response remains preserved.
    expected = {key: value for key, value in original.items() if key != "annotations"}
    assert calls[1]["messages"][2] == expected
    assert calls[1]["messages"][3] == {"role": "tool", "tool_call_id": "fixture-tool", "content": "file content"}
    assert calls[1]["max_tokens"] == 16384 and "temperature" not in calls[1]


def test_history_conversion_does_not_mutate_or_silently_drop_unknown_fields():
    from pydantic import ValidationError

    from benchmarks.ainsteinbench.harbor_native import completion_payload

    message = {
        "role": "assistant",
        "content": "answer",
        "annotations": [{"preserved": "raw trace"}],
        "reasoning_content": "unchanged",
    }
    original = copy.deepcopy(message)
    converted = completion_payload([message], creator_methods(SOURCE)["TOOLS"])
    assert message == original
    assert converted["messages"][0] == {key: value for key, value in original.items() if key != "annotations"}
    with pytest.raises(ValidationError):
        completion_payload([{**message, "unknown_response_extension": "reject"}], creator_methods(SOURCE)["TOOLS"])


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["cooldown", "response_read"])
async def test_agent_logical_deadline_covers_entire_shared_call(tmp_path, monkeypatch, phase):
    """Scale time only; retain the production 600-second boundary and one HTTP call."""
    import asyncio
    from unittest.mock import AsyncMock

    from harbor.models.agent.context import AgentContext

    from benchmarks.ainsteinbench import harbor_native as native

    requested_deadlines = []
    requests = []
    released = []
    cancelled = []

    def scaled_timeout(seconds):
        requested_deadlines.append(seconds)
        return asyncio.timeout(0.02)

    async def wait():
        try:
            await asyncio.sleep(1)
        finally:
            cancelled.append(phase)

    class Reply:
        def raise_for_status(self):
            pass

        async def json(self):
            await wait()

        def release(self):
            released.append(True)

    async def request(*args, **kwargs):
        requests.append(kwargs)
        if phase == "cooldown":
            await wait()
        return Reply()

    async def loop(**kwargs):
        return await kwargs["completion"](0, [{"role": "user", "content": "fixture"}], [])

    monkeypatch.setattr(native, "asyncio", SimpleNamespace(timeout=scaled_timeout))
    monkeypatch.setattr(native, "isolate", AsyncMock())
    monkeypatch.setattr(native, "creator_loop", loop)
    monkeypatch.setattr(native, "request", request)
    agent = object.__new__(native.AInsteinAgent)
    agent.source = SOURCE
    agent.logs_dir = tmp_path
    agent.endpoint = "http://127.0.0.1/ng-rollout/0-0"
    with pytest.raises(TimeoutError):
        await agent.run("MSB_pyscf_pyscf_pr2373", None, AgentContext())
    assert requested_deadlines == [600]
    assert len(requests) == 1
    assert requests[0]["_max_num_tries"] == 1
    assert requests[0]["_max_connection_retries"] == 0
    assert requests[0]["_resend_after_disconnect"] is False
    assert cancelled == [phase]
    assert released == ([True] if phase == "response_read" else [])
    assert not (tmp_path / "native/call-0.response.json").exists()


def test_et_prompt_uses_exact_creator_et_body_and_environment():
    from benchmarks.ainsteinbench.et_native import DESITTER
    from benchmarks.ainsteinbench.source import prompt

    rows, _ = checked_source(SOURCE)
    raw = rows[DESITTER]
    expected = creator_methods(SOURCE)["_build_et_prompt"](None, SimpleNamespace(**raw))
    assert prompt(SOURCE, raw) == expected
    assert "/opt/Cactus" in expected and "Thorn Name" in expected
    assert "PySCF" not in expected
    with pytest.raises(ValueError, match="Unsupported"):
        prompt(SOURCE, {**raw, "question_type": "unknown"})


def test_desitter_preparation_pins_qualified_image_and_explicit_placement(tmp_path):
    import tomllib

    from benchmarks.ainsteinbench.et_native import DESITTER, IMAGE

    rows = [json.loads(x) for x in stage(SOURCE, tmp_path).read_text().splitlines()]
    row = next(r for r in rows if r["task_id"] == DESITTER)
    assert row["verifier_metadata"]["blocked_reason"] is None
    config = tomllib.loads((tmp_path / "tasks" / DESITTER / "task.toml").read_text())
    assert config["environment"]["docker_image"] == IMAGE
    assert config["environment"]["cpus"] == 2
    assert config["environment"]["memory_mb"] == 4096
    assert config["verifier"]["environment_mode"] == "separate"
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert DESITTER in provenance["supported_native_tasks"]
    assert len(provenance["blocked_tasks"]) == 1


@pytest.mark.asyncio
async def test_et_original_build_failure_preserves_creator_score_and_command_bound(tmp_path):
    from benchmarks.ainsteinbench.et_native import DESITTER, IMAGE, score_et

    rows, _ = checked_source(SOURCE)
    commands = []
    uploads = []

    class Environment:
        task_env_config = SimpleNamespace(docker_image=IMAGE)

        async def upload_file(self, source, target):
            uploads.append(target)

        async def download_dir(self, source, target):
            target.mkdir(parents=True, exist_ok=True)

        async def exec(self, command, timeout_sec):
            commands.append((command, timeout_sec))
            return SimpleNamespace(
                return_code=1 if "simfactory/bin/sim build" in command else 0,
                stdout="compile failure" if "simfactory" in command else "",
                stderr="",
            )

    result = await score_et(SOURCE, Environment(), rows[DESITTER], "bad source", tmp_path)
    assert result["build"]["success"] is False
    assert result["native_score"]["overall_score"] == 0
    assert next(t for c, t in commands if "simfactory/bin/sim build" in c) == 300
    assert "/opt/Cactus/configs/sim/ThornList" in uploads
    assert "/opt/Cactus/arrangements/EinsteinInitialData/Exact/src/metrics/de_Sitter.F" in uploads
    assert not any("cactus_sim " in c for c, _ in commands)
    with pytest.raises(ValueError, match="not qualified"):
        other = Environment()
        other.task_env_config = SimpleNamespace(docker_image="old-image")
        await score_et(SOURCE, other, rows[DESITTER], "bad source", tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, 137, TimeoutError("transport timeout")])
async def test_et_transport_failure_never_becomes_creator_zero(tmp_path, failure):
    from benchmarks.ainsteinbench.et_native import DESITTER, IMAGE, score_et

    rows, _ = checked_source(SOURCE)

    class Environment:
        task_env_config = SimpleNamespace(docker_image=IMAGE)

        async def upload_file(self, source, target):
            pass

        async def download_dir(self, source, target):
            target.mkdir(parents=True, exist_ok=True)

        async def exec(self, command, timeout_sec):
            if "simfactory/bin/sim build" in command:
                if isinstance(failure, Exception):
                    raise failure
                return SimpleNamespace(return_code=failure, stdout="", stderr="")
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    with pytest.raises((RuntimeError, TimeoutError)):
        await score_et(SOURCE, Environment(), rows[DESITTER], "source", tmp_path)
    assert not (tmp_path / "score.json").exists()
    assert json.loads((tmp_path / "transport-status.json").read_text())["failures"]


@pytest.mark.asyncio
async def test_et_cancellation_settles_active_exec_before_return(tmp_path):
    import asyncio

    from benchmarks.ainsteinbench.et_native import DESITTER, IMAGE, score_et

    rows, _ = checked_source(SOURCE)
    started, release, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Environment:
        task_env_config = SimpleNamespace(docker_image=IMAGE)

        async def upload_file(self, source, target):
            pass

        async def download_dir(self, source, target):
            target.mkdir(parents=True, exist_ok=True)

        async def exec(self, command, timeout_sec):
            if "simfactory/bin/sim build" in command:
                assert timeout_sec == 300
                started.set()
                await release.wait()
                settled.set()
                return SimpleNamespace(return_code=1, stdout="compile failure", stderr="")
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    task = asyncio.create_task(score_et(SOURCE, Environment(), rows[DESITTER], "source", tmp_path))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert settled.is_set()
    assert json.loads((tmp_path / "transport-status.json").read_text())["worker_settled"]
    assert not (tmp_path / "score.json").exists()
