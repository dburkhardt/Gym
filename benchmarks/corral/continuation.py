# SPDX-License-Identifier: Apache-2.0
"""Opt-in ReAct decision-boundary snapshots in the creator's algorithm state.

This is not a process snapshot or automatic recovery of a terminal run. The caller
must preserve the isolated scientific environment and authoritative capture store.
"""

import asyncio
import copy
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import sys
import time
from pathlib import Path


NAMESPACE = "gym_react_provider_boundary_v1"
REACT_SHA256 = "e2d57a938b6ca26cbae5d2d9f6234a6c96ca6759dd16500975b1c36a870a0bb1"
MAX_CALLS = 50


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def projection(state):
    value = state.model_dump(mode="json")
    value.pop("through_commit_hash")
    for run in value["agent_runs"].values():
        run["algorithm_state"].pop(NAMESPACE, None)
    return digest(value)


def completed_calls(calls: Path, lineage: str, protocol: str) -> int:
    """Admit only the exact, contiguous, fully completed existing transport log."""
    if not calls.is_dir() or calls.is_symlink():
        raise ValueError("Missing Corral guarded call history")
    files = list(calls.iterdir())
    if any(not p.is_file() or p.is_symlink() for p in files):
        raise ValueError("Nonregular Corral call history")
    intents = sorted(calls.glob("*.intent.json"))
    expected = set()
    for ordinal, intent in enumerate(intents):
        stem = f"{ordinal:06d}"
        complete = calls / f"{stem}.complete.json"
        if intent.name != f"{stem}.intent.json" or not complete.is_file():
            raise ValueError("Corral provider outcome unresolved or ordinal gap")
        if json.loads(intent.read_text()) != {
            "logical_call": f"corral:{lineage}:{ordinal}",
            "config_digest": protocol,
        }:
            raise ValueError("Corral provider identity changed")
        if json.loads(complete.read_text()) != {"completed": True}:
            raise ValueError("Corral provider completion malformed")
        expected.update((intent.name, complete.name))
    if {p.name for p in files} != expected:
        raise ValueError("Corral call history contains failure or unknown evidence")
    return len(intents)


def implementation_binding():
    """Bind pinned creator closure plus actual installed dependency bytes."""
    from corral.agents.react import ReActAgent

    root = Path(inspect.getfile(ReActAgent)).resolve().parents[3]
    manifest = json.loads(Path(__file__).with_name("frozen_sources.json").read_text())
    for name, expected in manifest["files"].items():
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Corral continuation source changed: {name}")
    packages = {}
    for name in (
        "aiosqlite",
        "sqlalchemy",
        "pydantic",
        "rdkit",
        "numpy",
        "litellm",
        "promptstore",
        "openai",
        "openai-agents",
    ):
        distribution = importlib.metadata.distribution(name)
        files = {}
        installed_paths = set()
        for relative in distribution.files or ():
            if "__pycache__" in relative.parts or relative.suffix == ".pyc":
                continue
            path = Path(distribution.locate_file(relative))
            if path.is_file():
                files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
                installed_paths.add(path.resolve())
        if not files:
            raise ValueError(f"Corral dependency has no file inventory: {name}")
        module = "agents" if name == "openai-agents" else name
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None or Path(spec.origin).resolve() not in installed_paths:
            raise ValueError(f"Corral dependency import origin is unbound: {name}")
        packages[name] = {
            "version": distribution.version,
            "files_sha256": digest(files),
            "origin": str(Path(spec.origin).resolve()),
        }
    shared = {}
    for module in (
        "nemo_gym.base_responses_api_model",
        "nemo_gym._checkpoint.evaluation",
        "nemo_gym.episode_types",
        "nemo_gym.provider_dispatch",
    ):
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise ValueError(f"Missing Corral shared runtime: {module}")
        shared[module] = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
    return {
        "shared": shared,
        "creator_manifest": digest(manifest["files"]),
        "maintained": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("continuation.py", "guarded_agent.py")
        },
        "python": sys.version,
        "packages": packages,
    }


def capture_binding(capture_dir, episode, lineage, protocol, count):
    from nemo_gym._checkpoint.evaluation import capture_snapshot
    from nemo_gym.base_responses_api_model import CaptureStore

    snapshot = capture_snapshot([Path(capture_dir)], [episode])
    records = CaptureStore(capture_dir, provider_dispatch_guard=True).read(episode.capture_key)
    if len(records) != count:
        raise ValueError("Corral authoritative capture count disagrees with native state")
    for ordinal, record in enumerate(records):
        binding = record.get("provider_binding") or {}
        attempts = record.get("provider_attempts") or []
        if (
            binding.get("logical_call") != f"corral:{lineage}:{ordinal}"
            or binding.get("agent_config") != protocol
            or binding.get("logical_rollout") != episode.rollout_id
            or not attempts
            or any(a.get("outcome") != "captured" for a in attempts)
            or record.get("replay_response") is None
        ):
            raise ValueError("Corral authoritative provider binding is incomplete or changed")
    return snapshot


def _boundary(session, *, now, deadline):
    if not math.isfinite(deadline) or not math.isfinite(now) or now >= deadline:
        raise ValueError("Original Corral deadline expired or invalid")
    state = session.state
    run = state.agent_runs[session.actor.run_id]
    if run.status != "running" or state.submission is not None or state.runtime.status != "running":
        raise ValueError("Corral continuation requires a live nonterminal execution")
    if len(state.agent_runs) != 1:
        raise ValueError("Delegated Corral recovery is not qualified")
    if any(a.status in ("pending", "running") for a in state.actions.values()):
        raise ValueError("Pending Corral action must not be replayed")
    if any(t.status in ("pending", "running") for t in state.tool_invocations.values()):
        raise ValueError("Pending Corral tool invocation must not be replayed")
    if state.environment.values.get("jobs") or state.workspace.files:
        raise ValueError("Mutable workspace or background jobs are not qualified")
    count = state.usage.llm_calls
    if type(count) is not int or not 0 < count < MAX_CALLS:
        raise ValueError("Corral original interaction budget exhausted or unbound")
    if state.task.scaffold.get("max_iterations") != MAX_CALLS:
        raise ValueError("Corral original interaction budget changed")
    return count


async def save_boundary(
    session,
    messages,
    *,
    calls: Path,
    lineage: str,
    protocol: str,
    deadline: float,
    capture_dir: Path,
    episode,
    now=None,
):
    """Call while creator loop is paused before dispatch, after all tool results.

    Messages must be the actual ReAct local provider history, not reconstructed
    canonical session messages. Publication uses the creator commit API.
    """
    count = _boundary(session, now=time.time() if now is None else now, deadline=deadline)
    if completed_calls(calls, lineage, protocol) != count:
        raise ValueError("Corral native usage and guarded call history disagree")
    if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) for m in messages):
        raise ValueError("Missing exact Corral provider history")
    payload = {
        "version": 1,
        "state_digest": projection(session.state),
        "implementation": implementation_binding(),
        "captures": capture_binding(capture_dir, episode, lineage, protocol, count),
        "messages": copy.deepcopy(messages),
        "messages_sha256": digest(messages),
        "lineage": lineage,
        "protocol": protocol,
        "deadline": deadline,
        "next_ordinal": count,
        "original_max_calls": MAX_CALLS,
        "react_sha256": REACT_SHA256,
    }
    await session.set_agent_state(NAMESPACE, payload)
    return payload


def validate_boundary(
    session, *, calls: Path, lineage: str, protocol: str, deadline: float, capture_dir: Path, episode, now=None
):
    payload = session.get_agent_state(NAMESPACE)
    if not isinstance(payload, dict):
        raise ValueError("Missing exact Corral provider-history snapshot")
    count = _boundary(session, now=time.time() if now is None else now, deadline=deadline)
    if payload.get("version") != 1 or payload.get("original_max_calls") != MAX_CALLS:
        raise ValueError("Unsupported Corral continuation schema")
    expected = {
        "lineage": lineage,
        "protocol": protocol,
        "deadline": deadline,
        "next_ordinal": count,
        "react_sha256": REACT_SHA256,
    }
    if any(payload.get(k) != v for k, v in expected.items()):
        raise ValueError("Corral continuation identity or budget changed")
    if payload.get("implementation") != implementation_binding():
        raise ValueError("Corral continuation implementation or runtime changed")
    if payload.get("state_digest") != projection(session.state):
        raise ValueError("Corral state changed after supported boundary")
    if payload.get("messages_sha256") != digest(payload.get("messages")):
        raise ValueError("Corral provider history corrupted")
    if completed_calls(calls, lineage, protocol) != count:
        raise ValueError("Corral dispatch occurred after supported boundary")
    if payload.get("captures") != capture_binding(capture_dir, episode, lineage, protocol, count):
        raise ValueError("Corral capture evidence changed after supported boundary")
    return json.loads(json.dumps(payload, allow_nan=False))


def restored_agent(agent, payload):
    """Keep the pinned creator loop; change only initial history and remaining cap.

    Caller must first validate_boundary against the current native session and
    preserve the returned next_ordinal in the guarded transport. Not CLI admission.
    """
    from corral.agents.react import ReActAgent

    if (
        type(agent) is not ReActAgent
        or hashlib.sha256(Path(inspect.getfile(ReActAgent)).read_bytes()).hexdigest() != REACT_SHA256
    ):
        raise ValueError("Unqualified Corral agent source")
    if payload.get("react_sha256") != REACT_SHA256 or not 0 < payload.get("next_ordinal", 0) < MAX_CALLS:
        raise ValueError("Invalid Corral restored budget")

    class RemainingSession:
        def __init__(self, session):
            self._session = session

        def __getattr__(self, name):
            return getattr(self._session, name)

        @property
        def iteration_limit(self):
            return MAX_CALLS - payload["next_ordinal"]

    class RestoredReAct(ReActAgent):
        def _initial_messages(self, session):
            return json.loads(json.dumps(payload["messages"], allow_nan=False))

        async def run_session(self, session):
            remaining = payload["deadline"] - time.time()
            if remaining <= 0:
                raise ValueError("Original Corral deadline expired")
            async with asyncio.timeout(remaining):
                return await super().run_session(RemainingSession(session))

    restored = object.__new__(RestoredReAct)
    restored.__dict__.update(agent.__dict__)
    return restored
