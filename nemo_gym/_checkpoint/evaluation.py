# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable evaluation checkpoint publication; orchestration stays in coordination.

Unlike training, evaluation never falls back to restarting an un-restorable episode.
"""

import hashlib
import json
import os
from pathlib import Path

from nemo_gym._checkpoint import coordination
from nemo_gym.episode_types import EpisodeId
from nemo_gym.global_config import ConfigError


def file_digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ConfigError(f"Checkpoint requires an existing regular artifact: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_prefix(path: Path) -> dict:
    digest = file_digest(path)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": digest}


def validate_prefix(record: dict) -> None:
    path = Path(record["path"])
    file_digest(path)
    size = record["size"]
    if type(size) is not int or size < 0:
        raise ConfigError("Malformed checkpoint artifact prefix")
    with path.open("rb") as stream:
        prefix = stream.read(size)
    if len(prefix) != size or hashlib.sha256(prefix).hexdigest() != record["sha256"]:
        raise ConfigError(f"Checkpoint artifact history changed: {path}")


def expected_participants(config: dict) -> dict:
    kinds = {
        "environment_servers": "environment",
        "responses_api_models": "model",
        "responses_api_agents": "agent",
        "resources_servers": "resources",
    }
    result = {}
    for name, block in config.items():
        if not isinstance(block, dict):
            continue
        roles = [key for key in kinds if key in block]
        if len(roles) > 1:
            raise ConfigError("Checkpoint does not support ambiguous component roles")
        if roles:
            if not isinstance(block[roles[0]], dict) or len(block[roles[0]]) != 1:
                raise ConfigError("Malformed checkpoint component implementation")
            if roles[0] == "environment_servers" and "legacy_agent" in block[roles[0]]:
                # This inspected implementation is only an opaque HTTP relay.
                # Source/runtime binding happens before discovery; its target
                # must be the participating agent in the exact creator graph.
                from nemo_gym.creator_restart import exact_creator_graph

                graph = exact_creator_graph(config)
                if graph["relay"]["name"] != name:
                    raise ConfigError("Unqualified checkpoint relay")
                continue
            result[name] = kinds[roles[0]]
    if not result or "model" not in result.values():
        raise ConfigError("Checkpoint requires an explicit model participant")
    return result


def require_coverage(participants: coordination.Participants, expected: dict) -> None:
    actual = {member.server_name: member.kind for member in participants.members}
    if actual != expected:
        raise ConfigError("Checkpoint participant coverage differs from configured role graph")


def publish(path: Path, payload: dict) -> None:
    """Atomically publish only after all participant commits have succeeded."""
    if path.exists():
        raise ConfigError("Checkpoint publication already exists; use a fresh checkpoint directory")
    temporary = path.with_suffix(".pending")
    with temporary.open("x") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    # link is atomic and refuses a racing publication instead of replacing it.
    os.link(temporary, path)
    temporary.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_published(directory: Path) -> dict:
    path = directory / "evaluation.json"
    file_digest(path)
    try:
        data = json.loads(path.read_text())
        if data["version"] != 1 or not isinstance(data["binding"], dict):
            raise ValueError("unsupported receipt")
        ids = [EpisodeId.model_validate(value) for value in data["episodes"]]
        if not ids or len({e.rollout_id for e in ids}) != len(ids):
            raise ValueError("empty or duplicate episode scope")
        for record in data["prefixes"]:
            validate_prefix(record)
        for relative, digest in data["participant_files"].items():
            target = directory / relative
            if target.resolve().is_relative_to(directory.resolve()) is False or file_digest(target) != digest:
                raise ValueError("participant state changed")
        if not data["participant_files"]:
            raise ValueError("missing participant commits")
        return data
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ConfigError(f"Invalid published evaluation checkpoint: {error}") from error


def capture_snapshot(capture_dirs: list[Path], episodes: list[EpisodeId]) -> dict:
    """Bind all current attempts of selected logical rollouts; unknown sends refuse."""
    from nemo_gym.base_responses_api_model import CaptureStore

    logical = {episode.rollout_id for episode in episodes}
    result = {}
    for directory in capture_dirs:
        directory = Path(directory)
        identity_path = directory / ".capture-store-identity"
        file_digest(identity_path)
        identity = identity_path.read_text()
        import re

        if not re.fullmatch(r"[0-9a-f]{32}", identity):
            raise ConfigError("Missing or malformed original capture store identity")
        store = CaptureStore(directory, provider_dispatch_guard=True)
        store.assert_restore_safe([episode.capture_key for episode in episodes])
        files = {}
        for path in directory.iterdir():
            if not path.is_file():
                continue
            for suffix in (".capture.jsonl", ".capture.incomplete", ".capture.pending"):
                if path.name.endswith(suffix):
                    key = path.name.removesuffix(suffix)
                    if suffix == ".capture.pending":
                        key = key.rsplit(".", 1)[0]
                    if EpisodeId.from_capture_key(key).rollout_id in logical:
                        if store.is_incomplete(key) or store.has_pending(key):
                            raise ConfigError("Checkpoint has incomplete or pending model evidence")
                        files[path.name] = file_digest(path)
        result[str(directory.resolve())] = {"identity": identity, "files": files}
    if not result:
        raise ConfigError("Evaluation checkpoint requires durable model captures")
    return result


def validate_restore(directory: Path, *, binding: dict, capture_dirs: list[Path], pending_rows: list[dict]) -> dict:
    from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body

    data = read_published(directory)
    if data["binding"] != binding or not binding.get("restart_scope_qualified"):
        raise ConfigError("Checkpoint source/runtime/protocol binding changed or is unqualified")
    episodes = [EpisodeId.model_validate(value) for value in data["episodes"]]
    selected = {episode.rollout_id: episode for episode in episodes}
    pending = [EpisodeId.from_capture_key(maybe_rollout_id_from_run_body(row)) for row in pending_rows]
    if any(episode.rollout_id not in selected for episode in pending):
        raise ConfigError("Checkpoint restore has pending rows outside its exact scope")
    for episode in pending:
        if episode.attempt != selected[episode.rollout_id].attempt:
            raise ConfigError("Checkpoint restore attempt history changed")
    # Later completed rows may be retained, but never erase later evidence for a
    # still-pending episode. Conservative rejection is preferable to an invented replay.
    active = [selected[episode.rollout_id] for episode in pending]
    current = capture_snapshot(capture_dirs, active)
    saved = json.loads(json.dumps(data["captures"]))
    logical = {episode.rollout_id for episode in active}
    for store in saved.values():
        store["files"] = {
            name: sha
            for name, sha in store["files"].items()
            if EpisodeId.from_capture_key(
                name.removesuffix(".capture.jsonl").removesuffix(".capture.incomplete")
            ).rollout_id
            in logical
        }
    if active and current != saved:
        raise ConfigError("Checkpoint capture store changed after publication")
    return data


class EvaluationCheckpoint:
    """Collector-owned admission gate; uses existing participant coordination only."""

    def __init__(self, *, directory, client, config, binding, capture_dirs, artifacts):
        import asyncio

        self.directory = Path(directory)
        self.client = client
        self.config = config
        self.binding = binding
        self.capture_dirs = capture_dirs
        self.artifacts = artifacts
        self.admission = asyncio.Event()
        self.admission.set()
        self.requested = asyncio.Event()
        self.active = {}
        self.returned = set()
        self.persisted = set()
        self.error = None
        self.task = None
        self.participants = None

    async def before_dispatch(self, row):
        from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body

        await self.admission.wait()
        if self.error is not None:
            raise self.error
        key = maybe_rollout_id_from_run_body(row)
        self.active[key] = EpisodeId.from_capture_key(key)

    def after_reply(self, row):
        from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body

        self.returned.add(maybe_rollout_id_from_run_body(row))

    def after_persist(self, row):
        from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body

        key = maybe_rollout_id_from_run_body(row)
        self.persisted.add(key)
        self.active.pop(key, None)

    async def discover(self):
        token = self.config.get("checkpoint", {}).get("control_auth_token")
        if not token or not self.config.get("checkpoint", {}).get("enabled"):
            raise ConfigError("Evaluation checkpoints require enabled authenticated native checkpoint services")
        self.participants = await coordination.discover(self.client, auth_token=token)
        require_coverage(self.participants, expected_participants(self.config))

    async def checkpoint(self):
        import asyncio
        import time
        from uuid import uuid4

        self.admission.clear()
        identifier = uuid4().hex
        deadline = time.time() + 60
        try:
            await self.discover()
            prepared = await coordination.prepare(self.participants, identifier, deadline_ts=deadline)
            if not prepared.prepared or prepared.restarts():
                raise ConfigError("Evaluation checkpoint refuses non-quiescent or restart-only active episodes")
            # Replies already on the wire must reach the existing finalizer before
            # their output prefixes are published. Never cancel them into lost rows.
            while self.returned - self.persisted:
                if time.time() >= deadline:
                    raise ConfigError("Evaluation checkpoint could not drain completed rollout replies")
                await asyncio.sleep(0.01)
            episodes = list(self.active.values())
            if not episodes:
                raise ConfigError("No active episodes remain to checkpoint")
            captures = capture_snapshot(self.capture_dirs, episodes)
            self.directory.mkdir(parents=True, exist_ok=False)
            replies = await coordination.commit(
                self.participants, identifier, str(self.directory), episodes, deadline_ts=deadline
            )
            # Commit replies identify episodes still present after prepare. A
            # terminal /run reply can be in transport while no participant owns it.
            # Drain those excluded replies through the normal finalizer before publish.
            scopes = [
                set(replies[m.server_name]["episode_ids"])
                for m in self.participants.members
                if m.kind in {"environment", "agent"}
            ]
            if not scopes or any(scope != scopes[0] for scope in scopes):
                raise ConfigError("Checkpoint requires one qualified environment/agent routing scope")
            committed = scopes[0]
            excluded = {episode.capture_key for episode in episodes} - committed
            while excluded - self.persisted:
                if time.time() >= deadline:
                    raise ConfigError("Checkpoint could not drain replies outside participant commit scope")
                await asyncio.sleep(0.01)
            episodes = [episode for episode in episodes if episode.capture_key in committed]
            if not episodes:
                raise ConfigError("All episodes completed before checkpoint commit")
            captures = capture_snapshot(self.capture_dirs, episodes)
            prefixes = []
            for path in self.artifacts:
                with Path(path).open("rb") as stream:
                    os.fsync(stream.fileno())
                prefixes.append(artifact_prefix(Path(path)))
            participant_files = {
                str(path.relative_to(self.directory)): file_digest(path)
                for path in self.directory.rglob("*")
                if path.is_file()
            }
            publish(
                self.directory / "evaluation.json",
                {
                    "version": 1,
                    "checkpoint_id": identifier,
                    "binding": self.binding,
                    "episodes": [e.model_dump(mode="json") for e in episodes],
                    "prefixes": prefixes,
                    "participant_files": participant_files,
                    "captures": captures,
                    "participants": expected_participants(self.config),
                },
            )
        finally:
            import sys

            original_error = sys.exc_info()[1]
            try:
                if self.participants is not None:
                    resume_deadline = max(deadline, time.time() + 30)

                    async def finish_resume():
                        async with asyncio.timeout(max(0, resume_deadline - time.time())):
                            await coordination.resume(self.participants, identifier, deadline_ts=resume_deadline)

                    cleanup = asyncio.create_task(finish_resume())
                    cancelled = False
                    while not cleanup.done():
                        try:
                            await asyncio.shield(cleanup)
                        except asyncio.CancelledError:
                            cancelled = True
                    cleanup.result()
                    if cancelled:
                        raise asyncio.CancelledError()
            except BaseException as cleanup_error:
                if original_error is None:
                    raise
                original_error.add_note(
                    f"Checkpoint resume settlement: {type(cleanup_error).__name__}: {cleanup_error}"
                )
            finally:
                self.admission.set()

    async def watch(self):
        await self.requested.wait()
        try:
            await self.checkpoint()
        except Exception as error:
            self.error = error
            self.admission.set()
            raise

    async def restore(self, directory, pending_rows):
        import time

        from nemo_gym._checkpoint.control import next_attempt
        from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body

        data = validate_restore(
            Path(directory), binding=self.binding, capture_dirs=self.capture_dirs, pending_rows=pending_rows
        )
        await self.discover()
        if data["participants"] != expected_participants(self.config):
            raise ConfigError("Checkpoint role graph changed")
        pending_keys = {
            EpisodeId.from_capture_key(maybe_rollout_id_from_run_body(row)).rollout_id for row in pending_rows
        }
        episodes = [
            EpisodeId.model_validate(value) for value in data["episodes"] if value["rollout_id"] in pending_keys
        ]
        if not episodes:
            return
        # Preserve CoordinationError. Never fall back to input restart.
        await coordination.restore(
            self.participants, data["checkpoint_id"], str(directory), episodes, deadline_ts=time.time() + 60
        )
        await coordination.resume(self.participants, data["checkpoint_id"], deadline_ts=time.time() + 60)
        by_logical = {e.rollout_id: next_attempt(e) for e in episodes}
        for row in pending_rows:
            old = EpisodeId.from_capture_key(maybe_rollout_id_from_run_body(row))
            row["_ng_attempt_index"] = by_logical[old.rollout_id].attempt
