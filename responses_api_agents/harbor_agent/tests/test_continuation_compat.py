# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual pinned loop controls; local session is deliberately not native tmux."""

import copy
import hashlib
import json

import pytest


pytest.importorskip("harbor")
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.base import BaseLLM, LLMResponse
from harbor.models.agent.context import AgentContext

from responses_api_agents.harbor_agent.continuation_compat import install_continuation_hooks


class StopAtBoundary(BaseException):
    pass


class LocalModel(BaseLLM):
    def __init__(self):
        self.calls = []

    async def call(self, prompt, **kwargs):
        history = kwargs.get("message_history", [])
        self.calls.append(copy.deepcopy({"prompt": prompt, "history": history}))
        turn = len(history) // 2
        commands = [{"keystrokes": "write-a" if turn == 0 else "write-b", "duration": 0}] if turn < 2 else []
        return LLMResponse(
            content=json.dumps(
                {"analysis": "fixture", "plan": "fixture", "commands": commands, "task_complete": turn >= 2}
            )
        )

    def get_model_context_limit(self):
        return 100000

    def get_model_output_limit(self):
        return 1000


class LocalSession:
    """Synchronous file tools, fixed cwd/environment, no shell or background jobs."""

    def __init__(self, root):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.output = ""
        self.active = False

    async def attest_checkpoint_boundary(self, episode, prompt):
        if self.active or prompt != self.output:
            raise RuntimeError("Fixture tool state is not quiescent")
        if any(p.is_symlink() or not p.is_file() for p in self.root.iterdir()):
            raise RuntimeError("Fixture workspace contains unsupported state")

    async def is_session_alive(self):
        return True

    async def get_incremental_output(self):
        return self.output

    async def send_keys(self, keys, **kwargs):
        self.active = True
        try:
            if keys == "write-a":
                (self.root / "a").write_text("molecule-A\n")
            elif keys == "write-b":
                (self.root / "b").write_text((self.root / "a").read_text().upper())
            else:
                raise ValueError("Unsupported fixture tool")
            self.output = keys + " complete"
        finally:
            self.active = False


class LoopAgent(Terminus2):
    def _init_llm(self, *args, **kwargs):
        return LocalModel()

    async def _build_skills_section(self, environment):
        return None


class RestoredCursor:
    """Test-owned state handoff; NOT a production snapshot/restore implementation."""

    def __init__(self, agent, episode, prompt):
        assert not agent._session.active
        self.episode = episode
        self.prompt = prompt
        self.chat = {k: copy.deepcopy(v) for k, v in vars(agent._chat).items() if k != "_model"}
        self.agent = {
            k: copy.deepcopy(getattr(agent, k))
            for k in (
                "_pending_completion",
                "_trajectory_steps",
                "_api_request_times",
                "_n_episodes",
                "_summarization_count",
                "_subagent_metrics",
                "_subagent_rollout_details",
                "_pending_subagent_refs",
                "_pending_handoff_prompt",
                "_session_id",
            )
        }
        self.files = {p.name: p.read_bytes() for p in agent._session.root.iterdir()}
        self.digest = hashlib.sha256(b"".join(k.encode() + v for k, v in sorted(self.files.items()))).hexdigest()

    async def restore(self, agent, instruction, environment, context):
        from harbor.llms.chat import Chat

        if self.digest != hashlib.sha256(b"".join(k.encode() + v for k, v in sorted(self.files.items()))).hexdigest():
            raise ValueError("Corrupt fixture workspace")
        if type(self.episode) is not int or not 0 < self.episode < agent._max_episodes:
            raise ValueError("Invalid fixture cursor")
        if agent.options.enable_summarize:
            raise ValueError("Unsupported summarization")
        # Validate all before publishing files or modifying agent state.
        chat = Chat(agent._llm)
        for k, v in self.chat.items():
            setattr(chat, k, copy.deepcopy(v))
        for name, data in self.files.items():
            (agent._session.root / name).write_bytes(data)
        for k, v in self.agent.items():
            setattr(agent, k, copy.deepcopy(v))
        agent._chat = chat
        agent._context = context
        agent._gym_next_episode = self.episode
        return self.prompt


def agent(root):
    (root / "logs").mkdir(parents=True)
    result = LoopAgent(
        logs_dir=root / "logs",
        model_name="fixture",
        enable_summarize=False,
        max_turns=5,
        record_terminal_session=False,
    )
    result._session = LocalSession(root / "workspace")
    return result


@pytest.mark.asyncio
async def test_actual_loop_split_matches_uninterrupted(tmp_path):
    install_continuation_hooks(LoopAgent)
    uninterrupted = agent(tmp_path / "full")
    await uninterrupted.run("Transform a molecule file", None, AgentContext())
    first = agent(tmp_path / "split")
    saved = []

    async def boundary(owner, episode, prompt):
        saved.append(RestoredCursor(owner, episode, prompt))
        raise StopAtBoundary()

    first._gym_completed_turn = boundary
    with pytest.raises(StopAtBoundary):
        await first.run("Transform a molecule file", None, AgentContext())
    assert len(first._llm.calls) == 1
    restored = agent(tmp_path / "restored")
    restored._gym_continuation = saved[0]
    await restored.run("Transform a molecule file", None, AgentContext())
    assert restored._gym_continuation is None
    assert first._llm.calls + restored._llm.calls == uninterrupted._llm.calls
    assert restored._chat.messages == uninterrupted._chat.messages
    assert restored._n_episodes == uninterrupted._n_episodes
    assert (restored._session.root / "b").read_bytes() == (uninterrupted._session.root / "b").read_bytes()
    assert [s.step_id for s in restored._trajectory_steps] == [s.step_id for s in uninterrupted._trajectory_steps]


def test_patch_refuses_drift_atomically():
    class Changed(LoopAgent):
        async def _run_agent_loop(self, *args):
            pass

    original_run = Changed.run
    with pytest.raises(RuntimeError, match="source changed"):
        install_continuation_hooks(Changed)
    assert Changed.run is original_run
    with pytest.raises(ValueError, match="dedicated"):
        install_continuation_hooks(Terminus2)


@pytest.mark.asyncio
async def test_unattested_tool_session_cannot_checkpoint(tmp_path):
    install_continuation_hooks(LoopAgent)
    candidate = agent(tmp_path)
    candidate._session.attest_checkpoint_boundary = None

    async def never(*args):
        raise AssertionError("Snapshot must not be published")

    candidate._gym_completed_turn = never
    with pytest.raises(RuntimeError, match="cannot attest"):
        await candidate.run("Transform a molecule file", None, AgentContext())
    assert len(candidate._llm.calls) == 1


@pytest.mark.asyncio
async def test_corrupt_restore_refuses_before_workspace_install(tmp_path):
    install_continuation_hooks(LoopAgent)
    first = agent(tmp_path / "first")
    saved = []

    async def boundary(owner, episode, prompt):
        saved.append(RestoredCursor(owner, episode, prompt))
        raise StopAtBoundary()

    first._gym_completed_turn = boundary
    with pytest.raises(StopAtBoundary):
        await first.run("Transform a molecule file", None, AgentContext())
    saved[0].files["a"] = b"tampered"
    restored = agent(tmp_path / "restored")
    restored._gym_continuation = saved[0]
    with pytest.raises(ValueError, match="Corrupt"):
        await restored.run("Transform a molecule file", None, AgentContext())
    assert list(restored._session.root.iterdir()) == []
    assert restored._llm.calls == []


@pytest.mark.asyncio
async def test_exhausted_boundary_is_not_published_and_fresh_cursor_resets(tmp_path):
    install_continuation_hooks(LoopAgent)
    candidate = agent(tmp_path)
    candidate._max_episodes = 1
    candidate._gym_next_episode = 999

    async def never(*args):
        raise AssertionError("Exhausted turn is not a continuation")

    candidate._gym_completed_turn = never
    await candidate.run("Transform a molecule file", None, AgentContext())
    assert candidate._gym_next_episode == 0
    assert len(candidate._llm.calls) == 1


@pytest.mark.asyncio
async def test_unsupported_summarization_fails_before_any_model_call(tmp_path):
    install_continuation_hooks(LoopAgent)
    candidate = agent(tmp_path)
    candidate.options.enable_summarize = True
    candidate._gym_completed_turn = object()
    with pytest.raises(RuntimeError, match="summarization disabled"):
        await candidate.run("Transform a molecule file", None, AgentContext())
    assert candidate._llm.calls == []
