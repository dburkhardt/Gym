# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in completed-turn hooks for the exact reviewed Harbor pin.

This patch supplies a boundary, not a snapshot implementation. It is not installed
by the native agent. A controller must bind validated agent, tool, workspace and
remaining-budget state before using it; ordinary tmux does not attest quiescence.
"""

import asyncio
import hashlib
import inspect
import textwrap
from types import FunctionType


_SOURCE_HASHES = {
    "run": "ae6f884efdc175042067d2f7ce52b0fdbca4f50fa84b5473d91a29b33d1fadb8",
    "_run_agent_loop": "fb724c179afe13965defdd55f029e5da63b275075381744629a079b409d2feef",
}


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError("Pinned Harbor continuation patch context changed")
    return source.replace(old, new, 1)


def _patched_method(original: FunctionType, name: str) -> FunctionType:
    raw = inspect.getsource(original)
    if hashlib.sha256(raw.encode()).hexdigest() != _SOURCE_HASHES[name]:
        raise RuntimeError(f"Pinned Harbor {name} source changed; continuation refused")
    source = textwrap.dedent(raw)
    if name == "run":
        start = source.index("    self._reset_per_run_state()")
        end = source.index("    try:\n        await self._run_agent_loop(", start)
        setup = source[start:end]
        source = (
            source[:start]
            + (
                "    continuation = getattr(self, '_gym_continuation', None)\n"
                "    if (continuation is not None or getattr(self, '_gym_completed_turn', None) is not None) and self.options.enable_summarize:\n"
                "        raise RuntimeError('Continuation requires summarization disabled')\n"
                "    if continuation is None:\n"
                "        self._gym_next_episode = 0\n"
                "        self._gym_absolute_deadline = None\n" + textwrap.indent(setup, "    ") + "    else:\n"
                "        initial_prompt = await continuation.restore(self, instruction, environment, context)\n"
                "        self._gym_continuation = None\n"
                "\n"
            )
            + source[end:]
        )
        source = _replace_once(
            source,
            "        await self._run_agent_loop(\n"
            "            initial_prompt=initial_prompt,\n"
            "            chat=self._chat,\n"
            "            original_instruction=instruction,\n"
            "        )\n",
            "        deadline = getattr(self, '_gym_absolute_deadline', None)\n"
            "        remaining = None if deadline is None else max(0.0, deadline - time.time())\n"
            "        async with asyncio.timeout(remaining):\n"
            "            await self._run_agent_loop(\n"
            "                initial_prompt=initial_prompt,\n"
            "                chat=self._chat,\n"
            "                original_instruction=instruction,\n"
            "            )\n",
        )
    else:
        source = _replace_once(
            source,
            "    self._context.n_input_tokens = 0\n"
            "    self._context.n_output_tokens = 0\n"
            "    self._context.n_cache_tokens = 0\n"
            "    self._context.cost_usd = None\n",
            "    next_episode = getattr(self, '_gym_next_episode', 0)\n"
            "    if type(next_episode) is not int or not 0 <= next_episode < self._max_episodes:\n"
            "        raise ValueError('Invalid continuation episode cursor')\n"
            "    if next_episode == 0:\n"
            "        self._context.n_input_tokens = 0\n"
            "        self._context.n_output_tokens = 0\n"
            "        self._context.n_cache_tokens = 0\n"
            "        self._context.cost_usd = None\n",
        )
        source = _replace_once(
            source,
            "    for episode in range(self._max_episodes):",
            "    for episode in range(next_episode, self._max_episodes):",
        )
        source = _replace_once(
            source,
            "        # Dump trajectory after every episode to ensure we persist even if program terminates\n"
            "        self._dump_trajectory()\n",
            "        # Dump trajectory after every episode to ensure we persist even if program terminates\n"
            "        self._dump_trajectory()\n"
            "        checkpoint_hook = getattr(self, '_gym_completed_turn', None)\n"
            "        if checkpoint_hook is not None and episode + 1 < self._max_episodes and not (is_task_complete and was_pending_completion):\n"
            "            if self.options.enable_summarize or timeout_occurred:\n"
            "                raise RuntimeError('Continuation requires unsummarized completed tools')\n"
            "            attestor = getattr(self._session, 'attest_checkpoint_boundary', None)\n"
            "            if not callable(attestor):\n"
            "                raise RuntimeError('Tool session cannot attest checkpoint quiescence')\n"
            "            await attestor(episode + 1, observation)\n"
            "            await checkpoint_hook(self, episode + 1, observation)\n",
        )
    namespace = dict(original.__globals__)
    namespace["asyncio"] = asyncio
    exec(compile(source, f"<gym-pinned-harbor-{name}>", "exec"), namespace)
    result = namespace[name]
    result.__gym_continuation_original__ = original
    return result


def install_continuation_hooks(agent_class: type) -> None:
    """Patch a dedicated subclass atomically; never mutate the global upstream class.

    Both original methods must match the reviewed pin before either is installed.
    Restore is a trusted controller capability, consumed after successful install.
    It must validate the whole cursor and every external state before mutation.
    Restore and completed-turn callbacks are required to enforce workspace and
    external budget contracts. No callback means unchanged native execution.
    """
    from harbor.agents.terminus_2.terminus_2 import Terminus2

    if agent_class is Terminus2 or not issubclass(agent_class, Terminus2):
        raise ValueError("Continuation hooks require a dedicated Terminus2 subclass")
    methods = {}
    for name in _SOURCE_HASHES:
        current = getattr(agent_class, name)
        original = getattr(current, "__gym_continuation_original__", current)
        methods[name] = _patched_method(original, name)
    for name, method in methods.items():
        setattr(agent_class, name, method)
