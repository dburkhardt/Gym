# SPDX-License-Identifier: Apache-2.0
import json
import shutil

import pytest

from benchmarks.corral.qualify_native import ACTIONS, compare_state, fixture
from benchmarks.corral.state_artifact import pack_state
from nemo_gym.openai_utils import NeMoGymChatCompletion


def test_exact_fixture_protocol():
    for i, action in enumerate(ACTIONS):
        body = fixture(i)
        NeMoGymChatCompletion.model_validate(body)
        assert body["choices"][0]["message"]["content"] == action
    with pytest.raises(ValueError, match="exhausted"):
        fixture(2)


def test_full_state_loss_rejected(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    (original / "solver-result.json").write_text(json.dumps({"status": "submitted"}))
    state = original / "state/session"
    state.mkdir(parents=True)
    (state / "commits.sqlite3").write_bytes(b"creator persistence fixture")
    (state / "object.bin").write_bytes(b"scientific object fixture")
    manifest = pack_state(original, tmp_path / "state.tar")
    submitted = tmp_path / "submitted"
    shutil.copytree(original, submitted)
    assert len(compare_state(original, submitted, manifest)) == 3
    (submitted / "state/session/object.bin").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        compare_state(original, submitted, manifest)


def test_changed_state_bytes_and_extra_files_rejected(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    (original / "solver-result.json").write_text("{}")
    (original / "state").mkdir()
    (original / "state/commits.sqlite3").write_bytes(b"AAAA")
    manifest = pack_state(original, tmp_path / "state.tar")
    submitted = tmp_path / "submitted"
    shutil.copytree(original, submitted)
    (submitted / "state/commits.sqlite3").write_bytes(b"BBBB")
    with pytest.raises(ValueError, match="bytes"):
        compare_state(original, submitted, manifest)
    (submitted / "state/commits.sqlite3").write_bytes(b"AAAA")
    (submitted / "extra").write_text("unexpected")
    with pytest.raises(ValueError, match="unexpected"):
        compare_state(original, submitted, manifest)


def test_tool_error_or_wrong_task_cannot_pass_zero_score():
    from benchmarks.corral.qualify_native import validate_transcript

    first = [{"role": "user", "content": "exact native prompt"}]
    second = first + [
        {"role": "assistant", "content": ACTIONS[0]},
        {"role": "user", "content": "Observation: ['azo']", "name": "get_available_functional_groups"},
    ]
    calls = [{"messages": first}, {"messages": second}]
    submission = {"status": "submitted", "submission": "{}", "output": {"answer": "{}"}}
    validate_transcript(calls, "exact native prompt", "Observation: ['azo']", submission)
    with pytest.raises(ValueError, match="prompt"):
        validate_transcript(calls, "other task", "Observation: ['azo']", submission)
    second[-1]["content"] = "Observation: tool failed"
    with pytest.raises(ValueError, match="successful"):
        validate_transcript(calls, "exact native prompt", "Observation: ['azo']", submission)
    second[-1]["content"] = "Observation: ['azo']"
    submission["submission"] = "different answer"
    with pytest.raises(ValueError, match="Submission"):
        validate_transcript(calls, "exact native prompt", "Observation: ['azo']", submission)


@pytest.mark.asyncio
async def test_real_sandbox_download_facade_preserves_exact_file_destination(tmp_path):
    from unittest.mock import AsyncMock

    from nemo_gym.sandbox import AsyncSandbox

    provider = AsyncMock()
    handle = object()
    sandbox = AsyncSandbox(provider)
    sandbox._handle = handle
    sandbox._stopped = False
    destination = tmp_path / "export.tar"

    async def copy(actual_handle, source, target):
        assert actual_handle is handle
        assert source == "/tmp/export.tar"
        assert target == destination
        target.write_bytes(b"exact downloaded bundle bytes")

    provider.download_file.side_effect = copy
    await sandbox.download("/tmp/export.tar", str(destination))
    provider.download_file.assert_awaited_once_with(handle, "/tmp/export.tar", destination)
    assert destination.read_bytes() == b"exact downloaded bundle bytes"
