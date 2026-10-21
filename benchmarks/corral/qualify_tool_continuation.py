# SPDX-License-Identifier: Apache-2.0
"""Fresh offline creator-loop control with selected pinned RDKit tool execution.

The catalog/task are unchanged. Execution is restricted to two pure creator tool
closures; PostgreSQL, RXNMapper, full environment setup and scoring are not tested.
"""

import argparse
import ast
import asyncio
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from benchmarks.corral.continuation import save_boundary
from benchmarks.corral.guarded_agent import MODEL, guarded_agent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_tools(source):
    from loguru import logger
    from rdkit import Chem

    root = source / "tasks/retrosynthesis/retrosynthesis"
    namespace = {"Chem": Chem, "logger": logger}
    exec(compile((root / "constants.py").read_text(), str(root / "constants.py"), "exec"), namespace)
    for filename, names in [
        ("retrosynthesis_utils.py", {"detect_functional_groups_in_molecule", "get_functional_groups"}),
        ("tools.py", {"detect_functional_groups", "get_available_functional_groups"}),
    ]:
        tree = ast.parse((root / filename).read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in nodes} == names
        for node in nodes:
            node.decorator_list = []
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root / filename), "exec"), namespace)
    return {name: namespace[name] for name in ["detect_functional_groups", "get_available_functional_groups"]}


async def child(args):
    from corral.agents import base_agent
    from corral.agents import session as session_module
    from corral.agents.session import AgentSession
    from corral.core.actors import ActorRef
    from corral.core.commit import CommitRequest
    from corral.core.events import UsageDelta
    from corral.core.tool_catalog import ToolCatalogSnapshot
    from corral.core.transition import ToolEffects
    from corral.persistence.sqlite import SQLiteCommitStore

    from nemo_gym.base_responses_api_model import CaptureStore, _CaptureMiddleware
    from nemo_gym.episode_types import EpisodeId
    from nemo_gym.provider_dispatch import current_provider_dispatch

    config = json.loads((args.output / "control.json").read_text())
    store = SQLiteCommitStore(args.output / "state.sqlite3", execution_id=config["execution_id"])
    capture = CaptureStore(args.output / "capture", provider_dispatch_guard=True)
    capture.identity()
    episode = EpisodeId.from_capture_key("0-0")
    protocol = config["protocol"]
    lineage = config["lineage"]
    deadline = config["deadline"]
    if args.mode != "restore":
        connection = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro&immutable=1", uri=True)
        events = list(
            connection.execute("SELECT event_json,author_json FROM commits WHERE sequence<=2 ORDER BY sequence")
        )
        connection.close()
        parent = None
        for i, (event, actor) in enumerate(events):
            commit = await store.append(
                CommitRequest.model_validate(
                    {
                        "execution_id": config["execution_id"],
                        "branch_id": "main",
                        "request_id": f"fixture-initialize-{i}",
                        "based_on_hash": parent,
                        "author": json.loads(actor),
                        "event": json.loads(event),
                    }
                )
            )
            parent = commit.hash
    state = await store.materialize("main")
    run = next(iter(state.agent_runs.values()))
    environment = SimpleNamespace(
        workspace_path=None,
        current_task=SimpleNamespace(allow_previous_attempt_context=False),
        validate_state_tool_catalog=lambda s: ToolCatalogSnapshot.model_validate(s.task.environment["tool_catalog"]),
    )
    session = AgentSession(
        environment,
        state,
        actor=ActorRef(kind="agent", actor_id=run.actor_id, run_id=run.run_id),
        state_store=store,
        branch_id="main",
        max_iterations=50,
    )
    tools = selected_tools(args.source)

    def execute_selected(environment, state, action):
        if action.name not in tools:
            raise RuntimeError("Unqualified tool refused by offline selected-tool control")
        result = tools[action.name](**dict(action.arguments))
        with (args.output / "tools.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    {"name": action.name, "arguments": dict(action.arguments), "result": result, "pid": os.getpid()}
                )
                + "\n"
            )
        return ToolEffects(observation=str(result), status="success", usage_delta=UsageDelta(tool_calls=1))

    session_module.execute_action = execute_selected
    target = "COC(C1(C=C1[Si](C)(C)C)/C=C/C2=CC=CC=C2)=O"
    answers = [
        f"<action>detect_functional_groups</action><action_input>{json.dumps({'smiles': target})}</action_input>",
        "<action>get_available_functional_groups</action><action_input>{}</action_input>",
    ]

    async def provider(scope, receive, send):
        request = json.loads((await receive())["body"])
        headers = dict(scope["headers"])
        ordinal = int(headers[b"x-nemo-gym-logical-call-id"].decode().rsplit(":", 1)[1])
        assert ordinal in (0, 1)
        dispatch = current_provider_dispatch.get()
        dispatch.before_send()
        with (args.output / "requests.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    {
                        "ordinal": ordinal,
                        "request": request,
                        "headers": {k.decode(): v.decode() for k, v in scope["headers"]},
                        "pid": os.getpid(),
                    }
                )
                + "\n"
            )
        response = {
            "id": f"fixture-{ordinal}",
            "object": "chat.completion",
            "model": MODEL,
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": answers[ordinal]}}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        dispatch.response_headers(200)
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": json.dumps(response).encode()})

    middleware = _CaptureMiddleware(provider, store=capture, model_server_name="fixture", model_config_digest="b" * 64)

    async def fixture_llm(**kwargs):
        request = {
            "model": MODEL,
            "messages": kwargs["messages"],
            "temperature": kwargs["temperature"],
            "max_tokens": kwargs["max_tokens"],
            "stream": False,
        }
        scope = {
            "type": "http",
            "path": "/ng-rollout/0-0/v1/chat/completions",
            "method": "POST",
            "headers": [(k.encode(), v.encode()) for k, v in kwargs["extra_headers"].items()],
        }
        received = [{"type": "http.request", "body": json.dumps(request).encode()}]
        sent = []

        async def receive():
            return received.pop(0)

        async def send(message):
            sent.append(message)

        await middleware(scope, receive, send)
        response = json.loads(b"".join(m.get("body", b"") for m in sent))
        return SimpleNamespace(
            id=response["id"], content=response["choices"][0]["message"]["content"], usage=response["usage"]
        )

    base_agent.llm_call = fixture_llm

    class Done(BaseException):
        pass

    async def boundary(messages, ordinal):
        if ordinal == 2:
            raise Done()
        if args.mode == "interrupt" and ordinal == 1:
            await save_boundary(
                session,
                messages,
                calls=args.output / "calls",
                lineage=lineage,
                protocol=protocol,
                deadline=deadline,
                capture_dir=args.output / "capture",
                episode=episode,
            )
            (args.output / "checkpoint-ready.json").write_text(
                json.dumps({"pid": os.getpid(), "head": session.state.through_commit_hash})
            )
            await asyncio.Event().wait()

    kwargs = (
        {}
        if args.mode != "restore"
        else {
            "continuation_session": session,
            "capture_dir": args.output / "capture",
            "episode": episode,
            "deadline": deadline,
        }
    )
    try:
        with guarded_agent(
            "http://fixture/ng-rollout/0-0/v1",
            lineage,
            args.output / "calls",
            dispatch_digest=protocol,
            before_dispatch=boundary,
            **kwargs,
        ) as agent:
            try:
                await agent.run_session(session)
            except Done:
                pass
        assert session.state.usage.llm_calls == 2 and session.state.usage.tool_calls == 2
        (args.output / "terminal.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "usage": session.state.usage.model_dump(),
                    "deadline": deadline,
                    "calls": len(capture.read("0-0")),
                    "pending": capture.has_pending("0-0"),
                }
            )
        )
    finally:
        await store.aclose()


def main():
    parser = argparse.ArgumentParser()
    for name in ["source", "database", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=["baseline", "interrupt", "restore"])
    args = parser.parse_args()
    # These are in-process ASGI requests; no networking is needed, including import-time telemetry.
    sys.addaudithook(
        lambda event, details: (_ for _ in ()).throw(RuntimeError("network forbidden"))
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}
        else None
    )
    if args.mode:
        asyncio.run(child(args))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    before = sha(args.database)
    deadline = time.time() + 3600
    for name in ["baseline", "interrupted"]:
        out = args.output / name
        out.mkdir()
        (out / "control.json").write_text(
            json.dumps(
                {"execution_id": "offline-tool-" + name, "protocol": "a" * 64, "lineage": name, "deadline": deadline}
            )
        )
        command = [
            sys.executable,
            "-m",
            "benchmarks.corral.qualify_tool_continuation",
            "--source",
            str(args.source),
            "--database",
            str(args.database),
            "--output",
            str(out),
        ]
        if name == "baseline":
            with (out / "baseline.log").open("w") as log:
                subprocess.run(
                    command + ["--mode", "baseline"], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=60
                )
        else:
            with (out / "interrupt.log").open("w") as log:
                process = subprocess.Popen(command + ["--mode", "interrupt"], stdout=log, stderr=subprocess.STDOUT)
                try:
                    limit = time.monotonic() + 60
                    while not (out / "checkpoint-ready.json").exists():
                        if process.poll() is not None:
                            raise RuntimeError("checkpoint child failed before ready")
                        if time.monotonic() > limit:
                            raise TimeoutError("checkpoint publication")
                        time.sleep(0.05)
                    os.kill(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=10)
            with (out / "restore.log").open("w") as log:
                subprocess.run(
                    command + ["--mode", "restore"], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=60
                )

    def lines(name, file):
        return [json.loads(line) for line in (args.output / name / file).read_text().splitlines()]

    baseline = lines("baseline", "requests.jsonl")
    restored = lines("interrupted", "requests.jsonl")
    assert [r["ordinal"] for r in baseline] == [0, 1] == [r["ordinal"] for r in restored]
    assert [r["request"] for r in baseline] == [r["request"] for r in restored]
    for records, lineage in [(baseline, "baseline"), (restored, "interrupted")]:
        for ordinal, row in enumerate(records):
            assert row["headers"] == {
                "x-nemo-gym-logical-call-id": f"corral:{lineage}:{ordinal}",
                "x-nemo-gym-call-config-sha256": "a" * 64,
            }
    tools1 = lines("baseline", "tools.jsonl")
    tools2 = lines("interrupted", "tools.jsonl")
    assert [{k: v for k, v in r.items() if k != "pid"} for r in tools1] == [
        {k: v for k, v in r.items() if k != "pid"} for r in tools2
    ]
    assert restored[0]["pid"] != restored[1]["pid"] and sha(args.database) == before
    (args.output / "acceptance.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "scope": "selected pinned RDKit tools, creator loop/SQLite/AgentSession, guarded in-process ASGI capture",
                "external_calls": 0,
                "requests_per_execution": 2,
                "tools_per_execution": 2,
                "next_request_equal": True,
                "original_max_calls": 50,
                "absolute_deadline_preserved": deadline,
                "source_database_unchanged": before,
                "full_environment_or_canonical_scorer_qualified": False,
                "shared_cli_checkpoint_qualified": False,
                "helper_sha256": sha(Path(__file__).with_name("continuation.py")),
                "guard_sha256": sha(Path(__file__).with_name("guarded_agent.py")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
