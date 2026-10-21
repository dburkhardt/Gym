# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Subprocess driver using native Gym checkpoint hooks and pinned Terminus loop.

The synchronous file-tool session is a restricted qualification fixture, not a
production tmux or scientific container implementation.
"""

import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from harbor.models.agent.context import AgentContext

from nemo_gym._checkpoint.agent import AgentSessionParticipant
from nemo_gym._checkpoint.control import CheckpointRequest, CommitRequest, ParticipantControlPlane, RestoreRequest
from nemo_gym.episode_types import EpisodeId
from responses_api_agents.harbor_agent.continuation_compat import install_continuation_hooks
from responses_api_agents.harbor_agent.continuation_state import capture_turn_cursor, validate_turn_cursor
from responses_api_agents.harbor_agent.tests.test_continuation_compat import LoopAgent, agent


INSTRUCTION = "Transform a molecule file"


def digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


class Hooks:
    def __init__(self, root, deadline):
        self.root, self.deadline = root, deadline
        self.agents = {}
        self.states = {}
        self.restored = []

    async def export_agent_sessions(self, keys):
        return {key: self.states[key] for key in keys}

    async def restore_agent_sessions(self, sessions):
        self.root.mkdir(parents=True, exist_ok=True)
        stage_root = Path(tempfile.mkdtemp(prefix="owned-restore-", dir=self.root))
        staged = {}
        try:
            validated = []
            for index, session in enumerate(sessions):
                state = session.session
                files = state["files"]
                if (
                    not isinstance(files, dict)
                    or digest(files) != state["files_sha256"]
                    or any(
                        name not in {"a", "b", "effects"} or not isinstance(value, str)
                        for name, value in files.items()
                    )
                ):
                    raise ValueError("Invalid fixture workspace snapshot")
                candidate = agent(stage_root / str(index))
                cursor = validate_turn_cursor(
                    candidate,
                    state["cursor"],
                    binding=session.session_key,
                    instruction=INSTRUCTION,
                    absolute_deadline=self.deadline,
                )
                validated.append((session, candidate, cursor))
            for session, candidate, cursor in validated:
                for name, contents in session.session["files"].items():
                    (candidate._session.root / name).write_text(contents)

                class Restore:
                    async def restore(self, owner, instruction, environment, context, prepared=cursor):
                        return prepared.install(owner, context)

                candidate._gym_continuation = Restore()
                staged[session.session_key] = candidate
        except BaseException:
            shutil.rmtree(stage_root)
            raise
        # Atomic visible mapping publication after every immutable state is staged.
        self.agents = staged
        self.restored = sessions

    async def retire_agent_session(self, key):
        self.agents.pop(key, None)


async def run(mode, root):
    root.mkdir(parents=True, exist_ok=True)
    install_continuation_hooks(LoopAgent)
    metadata = root / "metadata.json"
    if mode == "initial":
        metadata.write_text(json.dumps({"deadline": time.time() + 120}))
    deadline = json.loads(metadata.read_text())["deadline"]
    hooks = Hooks(root / mode, deadline)
    participant = AgentSessionParticipant(hooks)
    controller = ParticipantControlPlane(participant, instance_name="terminus-fixture", lease_grace_seconds=60)
    episode = EpisodeId(rollout_id="fixture")
    checkpoint = str(root / "checkpoint")
    request = {"checkpoint_id": "c1", "deadline_ts": time.time() + 30}
    if mode == "full":
        candidate = agent(root / "full")
        context = AgentContext()
        await candidate.run(INSTRUCTION, None, context)
        (root / "full.json").write_text(
            json.dumps(
                {
                    "calls": candidate._llm.calls,
                    "messages": candidate._chat.messages,
                    "episodes": candidate._n_episodes,
                    "context": context.model_dump(mode="json"),
                }
            )
        )
        return
    if mode == "initial":
        candidate = agent(root / "initial" / "live")
        hooks.agents["session"] = candidate
        participant.open_session("session", episode)
        boundary_ready = asyncio.Event()
        boundary_release = asyncio.Event()

        async def execute():
            async with participant.activation("session", episode) as activation:

                async def boundary(owner, next_episode, prompt):
                    files = {p.name: p.read_text() for p in owner._session.root.iterdir()}
                    hooks.states["session"] = {
                        "cursor": capture_turn_cursor(
                            owner,
                            next_episode,
                            prompt,
                            binding="session",
                            instruction=INSTRUCTION,
                            absolute_deadline=deadline,
                        ),
                        "files": files,
                        "files_sha256": digest(files),
                    }
                    boundary_ready.set()
                    await boundary_release.wait()
                    await activation.boundary(lambda: {"next_episode": next_episode})

                candidate._gym_completed_turn = boundary
                await candidate.run(INSTRUCTION, None, AgentContext())

        task = asyncio.create_task(execute())
        await boundary_ready.wait()
        prepared = asyncio.create_task(controller.prepare(CheckpointRequest(**request)))
        await asyncio.sleep(0)
        boundary_release.set()
        result = await prepared
        if result["phase"] != "prepared":
            raise RuntimeError(result)
        committed = await controller.commit(CommitRequest(**request, checkpoint_dir=checkpoint, episode_ids=[episode]))
        (root / "initial-calls.json").write_text(json.dumps(candidate._llm.calls))
        (root / "ready.json").write_text(json.dumps({"pid": os.getpid(), "commit": committed}))
        # Parent SIGKILLs this exact process after durable native commit.
        await asyncio.Event().wait()
        await task
    else:
        restored = await controller.restore(
            RestoreRequest(**request, checkpoint_dir=checkpoint, episode_ids=[episode])
        )
        await controller.resume(CheckpointRequest(**request))
        session = hooks.restored[0]
        candidate = hooks.agents[session.session_key]
        context = AgentContext()
        async with participant.activation(session.session_key, session.episode_id):
            await candidate.run(INSTRUCTION, None, context)
        result = {
            "pid": os.getpid(),
            "context": context.model_dump(mode="json"),
            "restore": restored,
            "calls": candidate._llm.calls,
            "messages": candidate._chat.messages,
            "episodes": candidate._n_episodes,
            "files": {p.name: p.read_text() for p in candidate._session.root.iterdir()},
            "score_count": 1,
            "score": int((candidate._session.root / "b").read_text() == "MOLECULE-A\n"),
        }
        (root / "restored.json").write_text(json.dumps(result))


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1], Path(sys.argv[2])))
