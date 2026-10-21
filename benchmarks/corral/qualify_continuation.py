# SPDX-License-Identifier: Apache-2.0
"""Offline native-store/history control; no CLI or production restore acceptance.

The historical task is terminal. A new branch in a copy is used solely to test
creator serialization and the next request. No historical run is resumed.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from benchmarks.corral.continuation import NAMESPACE, REACT_SHA256, digest, projection, restored_agent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
        raise RuntimeError("Offline Corral control forbids network")


async def child(args):
    from corral.agents import react
    from corral.agents.react import ReActAgent
    from corral.agents.session import AgentSession
    from corral.core.actors import ActorRef
    from corral.core.tool_catalog import ToolCatalogSnapshot
    from corral.persistence.sqlite import SQLiteCommitStore

    receipt = json.loads((args.output / "state-binding.json").read_text())
    store = SQLiteCommitStore(args.output / "commits.sqlite3", execution_id=receipt["execution_id"])
    try:
        if args.mode == "save":
            await store.create_branch(branch_id="offline-boundary-control", from_hash=receipt["head"])
        state = await store.materialize("offline-boundary-control")
        run = next(iter(state.agent_runs.values()))
        # No environment/tool execution is permitted by this read-only context facade.
        environment = SimpleNamespace(
            workspace_path=None,
            current_task=SimpleNamespace(allow_previous_attempt_context=False),
            validate_state_tool_catalog=lambda s: ToolCatalogSnapshot.model_validate(
                s.task.environment["tool_catalog"]
            ),
        )
        session = AgentSession(
            environment,
            state,
            actor=ActorRef(kind="agent", actor_id=run.actor_id, run_id=run.run_id),
            state_store=store,
            branch_id="offline-boundary-control",
            max_iterations=50,
        )
        request = json.loads(args.request.read_text())
        agent = ReActAgent(
            model="openai/" + request["model"], temperature=request["temperature"], max_tokens=request["max_tokens"]
        )
        native_messages = agent._initial_messages(session)
        if args.mode == "save":
            assert native_messages != request["messages"], "Expected pinned creator history discrepancy disappeared"
            deadline = state.runtime.started_at.timestamp() + 3600
            payload = {
                "version": 1,
                "state_digest": projection(state),
                "messages": request["messages"],
                "messages_sha256": digest(request["messages"]),
                "next_ordinal": state.usage.llm_calls,
                "original_max_calls": 50,
                "react_sha256": REACT_SHA256,
                "deadline": deadline,
            }
            # This control seeds native algorithm state from saved evidence. It does not
            # fabricate a qualified capture-store receipt or call production admission.
            await session.set_agent_state(NAMESPACE, payload)
            (args.output / "saved.json").write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "head": session.state.through_commit_hash,
                        "native_message_count": len(native_messages),
                        "expected_message_count": len(request["messages"]),
                        "tool_observation": list(state.tool_invocations.values())[-1].observation,
                        "state_digest": payload["state_digest"],
                    },
                    indent=2,
                )
            )
        else:
            payload = session.get_agent_state(NAMESPACE)
            assert payload["state_digest"] == projection(state)
            restored = restored_agent(agent, payload)
            observed = []

            class ReachedBoundary(BaseException):
                pass

            async def stop_before_provider(**kwargs):
                observed.append(kwargs)
                raise ReachedBoundary()

            original = react.call_model
            react.call_model = stop_before_provider
            # Only this offline control moves the clock back into the historical window.
            import benchmarks.corral.continuation as boundary

            original_clock = boundary.time.time
            boundary.time.time = lambda: payload["deadline"] - 1200
            try:
                try:
                    await restored.run_session(session)
                except ReachedBoundary:
                    pass
            finally:
                react.call_model = original
                boundary.time.time = original_clock
            assert len(observed) == 1
            actual = observed[0]
            assert actual["messages"] == request["messages"]
            assert actual["model"] == "openai/" + request["model"]
            assert actual["temperature"] == request["temperature"]
            assert actual["kwargs"]["max_tokens"] == request["max_tokens"]
            assert state.usage.llm_calls == 13
            (args.output / "restored.json").write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "history_exact": True,
                        "next_ordinal": 13,
                        "remaining_original_calls": 37,
                        "absolute_deadline": payload["deadline"],
                        "new_provider_calls": 0,
                        "new_tool_calls": 0,
                        "virtual_historical_clock": True,
                        "request_sha256": sha(args.request),
                        "request_messages_sha256": digest(actual["messages"]),
                    },
                    indent=2,
                )
            )
    finally:
        await store.aclose()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["save", "restore"])
    args = parser.parse_args()
    sys.addaudithook(deny_network)
    if args.mode:
        asyncio.run(child(args))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    before = sha(args.database)
    source = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro&immutable=1", uri=True)
    target = sqlite3.connect(args.output / "commits.sqlite3")
    source.backup(target)
    target.close()
    row = source.execute("SELECT hash, execution_id, event_type FROM commits WHERE sequence=33").fetchone()
    source.close()
    assert row[2] == "tool.completed"
    (args.output / "state-binding.json").write_text(json.dumps({"head": row[0], "execution_id": row[1]}))
    command = [
        sys.executable,
        "-m",
        "benchmarks.corral.qualify_continuation",
        "--database",
        str(args.database),
        "--request",
        str(args.request),
        "--output",
        str(args.output),
    ]
    for mode in ["save", "restore"]:
        with (args.output / f"{mode}.log").open("w") as log:
            subprocess.run(command + ["--mode", mode], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=60)
    saved = json.loads((args.output / "saved.json").read_text())
    restored = json.loads((args.output / "restored.json").read_text())
    assert saved["pid"] != restored["pid"] and sha(args.database) == before
    (args.output / "acceptance.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "scope": "native SQLite/AgentSession provider-history restoration only",
                "historical_terminal_run_resumed": False,
                "capture_admission_qualified": False,
                "saved_database": str(args.database.resolve()),
                "saved_database_sha256": before,
                "request": str(args.request.resolve()),
                "request_sha256": sha(args.request),
                "helper_sha256": sha(Path(__file__).with_name("continuation.py")),
                "control_sha256": sha(__file__),
                "saved": saved,
                "restored": restored,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
