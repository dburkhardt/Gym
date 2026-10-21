# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent rejection controls; no model or server dispatch."""

import copy
import json
from unittest.mock import AsyncMock

import pytest

from nemo_gym._checkpoint import coordination, evaluation
from nemo_gym.episode_types import EpisodeId
from nemo_gym.global_config import ConfigError


def tree(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


@pytest.fixture
def published(tmp_path):
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / ".capture-store-identity").write_text("a" * 32)
    output = tmp_path / "results.jsonl"
    output.write_bytes(b'{"retained": true}\n')
    state = directory / "agent.json"
    state.write_text("{}")
    binding = {"restart_scope_qualified": True, "source": "original"}
    payload = {
        "version": 1,
        "checkpoint_id": "checkpoint1",
        "binding": binding,
        "episodes": [{"rollout_id": "task1", "attempt": 0}],
        "prefixes": [evaluation.artifact_prefix(output)],
        "participant_files": {"agent.json": evaluation.file_digest(state)},
        "captures": evaluation.capture_snapshot([capture], [EpisodeId(rollout_id="task1")]),
        "participants": {"model": "model", "agent": "agent"},
    }
    evaluation.publish(directory / "evaluation.json", payload)
    return directory, capture, binding, [{"_ng_rollout_id": "task1", "_ng_attempt_index": 0}]


@pytest.mark.parametrize("existing_directory", [False, True])
def test_missing_capture_identity_does_not_create_files(tmp_path, existing_directory):
    capture = tmp_path / "absent"
    if existing_directory:
        capture.mkdir()
    before = tree(tmp_path)
    with pytest.raises(ConfigError):
        evaluation.capture_snapshot([capture], [EpisodeId(rollout_id="task1")])
    assert tree(tmp_path) == before
    assert capture.exists() is existing_directory


def test_changed_binding_and_truncated_tail_are_preserved(published, tmp_path):
    directory, capture, binding, rows = published
    with (tmp_path / "results.jsonl").open("ab") as stream:
        stream.write(b'{"unfinished":')
    before = tree(tmp_path)
    with pytest.raises(ConfigError, match="binding changed"):
        evaluation.validate_restore(
            directory, binding={**binding, "source": "changed"}, capture_dirs=[capture], pending_rows=rows
        )
    assert tree(tmp_path) == before


def test_corrupt_participant_refuses_without_mutation(published, tmp_path):
    directory, capture, binding, rows = published
    (directory / "agent.json").write_text('{"corrupt":true}')
    before = tree(tmp_path)
    with pytest.raises(ConfigError, match="participant state changed"):
        evaluation.validate_restore(directory, binding=binding, capture_dirs=[capture], pending_rows=rows)
    assert tree(tmp_path) == before


def test_late_unknown_attempt_refuses_without_mutation(published, tmp_path):
    directory, capture, binding, rows = published
    (capture / "task1-a2.call.capture.pending").write_text(
        json.dumps({"model_call_id": "call", "provider_attempts": [{"outcome": "unknown"}]})
    )
    before = tree(tmp_path)
    with pytest.raises(Exception, match="unresolved|pending|unknown"):
        evaluation.validate_restore(directory, binding=binding, capture_dirs=[capture], pending_rows=rows)
    assert tree(tmp_path) == before


@pytest.mark.parametrize("row", [{"_ng_rollout_id": "other"}, {"_ng_rollout_id": "task1", "_ng_attempt_index": 1}])
def test_exact_episode_scope_required(published, row):
    directory, capture, binding, _ = published
    with pytest.raises(ConfigError, match="scope|attempt history"):
        evaluation.validate_restore(directory, binding=binding, capture_dirs=[capture], pending_rows=[row])


def test_auxiliary_model_cannot_disappear_from_coverage():
    participants = coordination.Participants(None, "fixture", (coordination.Participant("model", "model"),))
    with pytest.raises(ConfigError, match="coverage"):
        evaluation.require_coverage(participants, {"model": "model", "judge": "model"})


@pytest.mark.asyncio
async def test_partial_restore_failure_never_advances_pending_attempt(published, tmp_path, monkeypatch):
    directory, capture, binding, rows = published
    original = copy.deepcopy(rows)
    config = {
        "model": {"responses_api_models": {"openai_model": {}}},
        "agent": {"responses_api_agents": {"simple_agent": {}}},
    }
    checkpoint = evaluation.EvaluationCheckpoint(
        directory=tmp_path / "unused",
        client=None,
        config=config,
        binding=binding,
        capture_dirs=[capture],
        artifacts=[],
    )
    checkpoint.discover = AsyncMock()
    restore = AsyncMock(side_effect=coordination.CoordinationError("restore", {"agent": "fixture failure"}))
    resume = AsyncMock()
    monkeypatch.setattr(coordination, "restore", restore)
    monkeypatch.setattr(coordination, "resume", resume)
    before = tree(tmp_path)
    with pytest.raises(coordination.CoordinationError):
        await checkpoint.restore(directory, rows)
    assert rows == original
    assert tree(tmp_path) == before
    resume.assert_not_awaited()
    assert checkpoint.active == {}


def test_two_attempts_of_same_logical_episode_refused(published, tmp_path):
    directory, capture, binding, rows = published
    receipt = directory / "evaluation.json"
    data = json.loads(receipt.read_text())
    data["episodes"].append({"rollout_id": "task1", "attempt": 1})
    receipt.write_text(json.dumps(data))
    before = tree(tmp_path)
    with pytest.raises(ConfigError, match="duplicate|scope|attempt"):
        evaluation.read_published(directory)
    assert tree(tmp_path) == before
