# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual pinned loop controls; local session is deliberately not native tmux."""

import copy
import hashlib
import json
import time

import pytest


pytest.importorskip("harbor")
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.base import BaseLLM, LLMResponse
from harbor.models.agent.context import AgentContext
from harbor.models.metric import UsageInfo

from responses_api_agents.harbor_agent.continuation_compat import install_continuation_hooks
from responses_api_agents.harbor_agent.continuation_state import capture_turn_cursor, validate_turn_cursor


class StopAtBoundary(BaseException):
    pass


class LocalModel(BaseLLM):
    def __init__(self):
        self.calls = []

    async def call(self, prompt, **kwargs):
        history = kwargs.get("message_history", [])
        self.calls.append(
            copy.deepcopy(
                {
                    "prompt": prompt,
                    "history": history,
                    "settings": {k: v for k, v in kwargs.items() if k not in {"message_history", "logging_path"}},
                }
            )
        )
        turn = len(history) // 2
        commands = [{"keystrokes": "write-a" if turn == 0 else "write-b", "duration": 0}] if turn < 2 else []
        return LLMResponse(
            usage=UsageInfo(prompt_tokens=7, completion_tokens=3, cache_tokens=1, cost_usd=0.001),
            content=json.dumps(
                {"analysis": "fixture", "plan": "fixture", "commands": commands, "task_complete": turn >= 2}
            ),
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
            with (self.root / "effects").open("a") as handle:
                handle.write(keys + "\n")
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
        self.deadline = time.time() + 60
        self.cursor = capture_turn_cursor(
            agent,
            episode,
            prompt,
            binding="fixture-task",
            instruction="Transform a molecule file",
            absolute_deadline=self.deadline,
        )
        self.files = {p.name: p.read_bytes() for p in agent._session.root.iterdir()}
        self.digest = hashlib.sha256(b"".join(k.encode() + v for k, v in sorted(self.files.items()))).hexdigest()

    async def restore(self, agent, instruction, environment, context):
        if self.digest != hashlib.sha256(b"".join(k.encode() + v for k, v in sorted(self.files.items()))).hexdigest():
            raise ValueError("Corrupt fixture workspace")
        validated = validate_turn_cursor(
            agent,
            json.loads(json.dumps(self.cursor)),
            binding="fixture-task",
            instruction=instruction,
            absolute_deadline=self.deadline,
        )
        for name, data in self.files.items():
            (agent._session.root / name).write_bytes(data)
        return validated.install(agent, context)


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


def test_native_gym_checkpoint_clean_process_restore(tmp_path):
    import os
    import subprocess
    import sys

    env = {**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
    command = [sys.executable, "-m", "responses_api_agents.harbor_agent.tests.continuation_process_control"]
    log = (tmp_path / "initial.log").open("w")
    process = subprocess.Popen(command + ["initial", str(tmp_path)], env=env, stdout=log, stderr=log)
    try:
        deadline = time.monotonic() + 20
        while not (tmp_path / "ready.json").exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise AssertionError((tmp_path / "initial.log").read_text())
            time.sleep(0.05)
        ready = json.loads((tmp_path / "ready.json").read_text())
        assert ready["pid"] == process.pid
        process.kill()
        process.wait(timeout=5)
        completed = subprocess.run(
            command + ["restore", str(tmp_path)], env=env, capture_output=True, text=True, timeout=20
        )
        assert completed.returncode == 0, completed.stderr
        result = json.loads((tmp_path / "restored.json").read_text())
        assert result["pid"] != ready["pid"]
        assert result["files"]["effects"].splitlines() == ["write-a", "write-b"]
        assert result["score_count"] == result["score"] == 1
        first = json.loads((tmp_path / "initial-calls.json").read_text())
        assert len(first) == 1 and len(result["calls"]) == 3
        assert result["calls"][0]["prompt"] == "write-a complete"
        assert len(result["calls"][0]["history"]) == 2
        full_process = subprocess.run(
            command + ["full", str(tmp_path)], env=env, capture_output=True, text=True, timeout=20
        )
        assert full_process.returncode == 0, full_process.stderr
        full = json.loads((tmp_path / "full.json").read_text())
        assert first + result["calls"] == full["calls"]
        assert result["messages"] == full["messages"]
        assert result["episodes"] == full["episodes"]
        for key in ("n_input_tokens", "n_output_tokens", "n_cache_tokens", "cost_usd"):
            assert result["context"][key] == full["context"][key]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        log.close()


@pytest.mark.asyncio
async def test_deadline_and_usage_survive_staging_and_early_timeout(tmp_path, monkeypatch):
    import asyncio

    import responses_api_agents.harbor_agent.continuation_state as state_module

    install_continuation_hooks(LoopAgent)
    first = agent(tmp_path / "first")
    saved = []

    async def boundary(owner, episode, prompt):
        saved.append((owner, episode, prompt))
        raise StopAtBoundary()

    first._gym_completed_turn = boundary
    with pytest.raises(StopAtBoundary):
        await first.run("Transform a molecule file", None, AgentContext())
    owner, episode, prompt = saved[0]
    deadline = time.time() + 0.2
    cursor = capture_turn_cursor(
        owner, episode, prompt, binding="task", instruction="Transform a molecule file", absolute_deadline=deadline
    )
    restored = agent(tmp_path / "restored")
    prepared = validate_turn_cursor(
        restored, cursor, binding="task", instruction="Transform a molecule file", absolute_deadline=deadline
    )
    with pytest.raises(ValueError, match="changed or expired"):
        validate_turn_cursor(
            restored, cursor, binding="task", instruction="Transform a molecule file", absolute_deadline=deadline + 60
        )
    context = AgentContext()

    class Restore:
        async def restore(self, *args):
            return prepared.install(restored, context)

    restored._gym_continuation = Restore()

    async def never_ready():
        await asyncio.Event().wait()

    restored._session.is_session_alive = never_ready
    with pytest.raises(TimeoutError):
        await restored.run("Transform a molecule file", None, context)
    assert restored._llm.calls == []
    assert (context.n_input_tokens, context.n_output_tokens, context.n_cache_tokens) == (7, 3, 1)
    assert context.cost_usd == 0.001
    before = restored._chat
    with monkeypatch.context() as patch:
        patch.setattr(state_module.time, "time", lambda: deadline + 1)
        with pytest.raises(ValueError, match="expired during staging"):
            prepared.install(restored, AgentContext())
    assert restored._chat is before


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["invalid_second", "write_second"])
async def test_two_session_restore_failure_discards_staging(tmp_path, monkeypatch, failure):
    from pathlib import Path

    from nemo_gym._checkpoint.agent import RestoredAgentSession
    from nemo_gym.episode_types import EpisodeId
    from responses_api_agents.harbor_agent.tests.continuation_process_control import Hooks, digest

    install_continuation_hooks(LoopAgent)
    first = agent(tmp_path / "source")
    saved = []

    async def boundary(owner, episode, prompt):
        saved.append((owner, episode, prompt))
        raise StopAtBoundary()

    first._gym_completed_turn = boundary
    with pytest.raises(StopAtBoundary):
        await first.run("Transform a molecule file", None, AgentContext())
    owner, episode, prompt = saved[0]
    deadline = time.time() + 60
    files = {p.name: p.read_text() for p in owner._session.root.iterdir()}
    sessions = []
    for key in ("one", "two"):
        cursor = capture_turn_cursor(
            owner, episode, prompt, binding=key, instruction="Transform a molecule file", absolute_deadline=deadline
        )
        sessions.append(
            RestoredAgentSession(
                session_key=key,
                episode_id=EpisodeId(rollout_id=key),
                session={"cursor": cursor, "files": files, "files_sha256": digest(files)},
            )
        )
    hooks = Hooks(tmp_path / "staging", deadline)
    original = agent(tmp_path / "original")
    (original._session.root / "a").write_text("preserved")
    hooks.agents = {"original": original}
    if failure == "invalid_second":
        sessions[1].session["files_sha256"] = "wrong"
    else:
        native_write = Path.write_text

        def fail_second(path, *args, **kwargs):
            if "owned-restore-" in str(path) and path.parts[-3:] == ("1", "workspace", "a"):
                raise OSError("Injected second workspace write failure")
            return native_write(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_second)
    with pytest.raises((ValueError, OSError)):
        await hooks.restore_agent_sessions(sessions)
    assert hooks.agents == {"original": original}
    assert (original._session.root / "a").read_text() == "preserved"
    assert list((tmp_path / "staging").glob("owned-restore-*")) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("contradiction", ["reasoning", "trajectory", "effective_kwargs", "instruction"])
async def test_self_consistent_but_incompatible_cursor_refused(tmp_path, contradiction):
    from responses_api_agents.harbor_agent.continuation_state import _digest

    install_continuation_hooks(LoopAgent)
    first = agent(tmp_path / "first")
    saved = []

    async def boundary(owner, episode, prompt):
        saved.append(RestoredCursor(owner, episode, prompt))
        raise StopAtBoundary()

    first._gym_completed_turn = boundary
    with pytest.raises(StopAtBoundary):
        await first.run("Transform a molecule file", None, AgentContext())
    saved = saved[0]
    candidate = agent(tmp_path / "candidate")
    instruction = "Transform a molecule file"
    if contradiction == "reasoning":
        saved.cursor["state"]["chat"]["_interleaved_thinking"] = True
    elif contradiction == "trajectory":
        saved.cursor["state"]["trajectory"] = saved.cursor["state"]["trajectory"][:1]
    elif contradiction == "effective_kwargs":
        candidate._llm_call_kwargs["temperature"] = 0.85
    else:
        instruction = "Different scientific task"
    saved.cursor["sha256"] = _digest(saved.cursor["state"])
    with pytest.raises(ValueError):
        validate_turn_cursor(
            candidate, saved.cursor, binding="fixture-task", instruction=instruction, absolute_deadline=saved.deadline
        )
    assert candidate._chat is None
    assert candidate._llm.calls == []
    assert list(candidate._session.root.iterdir()) == []
