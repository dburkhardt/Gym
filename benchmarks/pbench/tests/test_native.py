# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest
from harbor.models.task.config import TaskConfig

from benchmarks.pbench.managed_regrade import validate_workspace
from benchmarks.pbench.native_prepare import IMAGE, native_prepare


SOURCE = Path(__file__).resolve().parents[5] / "outputs/broad-eval-campaign/harbor/agent-expansion/pbench-harbor"


def test_native_preparation_preserves_science_and_pins_runtime(tmp_path):
    if not SOURCE.is_dir():
        pytest.skip("Preserved campaign source unavailable")
    output = native_prepare(SOURCE, tmp_path / "native", 5)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 5
    manifest = json.loads(Path(__file__).resolve().parents[1].joinpath("frozen_sources.json").read_text())
    for task in manifest["tasks"][:5]:
        name = task["harbor_task"]
        staged = output.parent / "tasks" / name
        original = SOURCE / "tasks" / name
        for relative in task["files"]:
            if relative not in ("environment/Dockerfile", "task.toml"):
                assert (staged / relative).read_bytes() == (original / relative).read_bytes()
        assert (staged / "environment/Dockerfile").read_text().startswith(f"FROM {IMAGE}\n")
        assert TaskConfig.model_validate(
            __import__("tomllib").loads((staged / "task.toml").read_text())
        ).artifacts == ["/workspace"]
        assert not (staged / "environment" / "answer_key.json").exists()
    assert [r["task_id"] for r in rows] == ["pbench::" + t["harbor_task"] for t in manifest["tasks"][:5]]


def test_missing_result_needs_explicit_complete_capture(tmp_path):
    task_id = "t_test--dfeep__paper_93__task_7"
    with pytest.raises(ValueError, match="capture evidence"):
        validate_workspace(task_id, tmp_path)
    # An empty but completely captured workspace may be a genuine model omission.
    assert validate_workspace(task_id, tmp_path, capture_complete=True)["harbor_task"] == task_id
    (tmp_path / "result.json").symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError, match="symlinks"):
        validate_workspace(task_id, tmp_path, capture_complete=True)


@pytest.mark.asyncio
async def test_failed_managed_scorer_is_not_a_zero_and_cleanup_settles(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from benchmarks.pbench import managed_regrade as verifier

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    state = {"closed": False}

    class Box:
        async def exec(self, *args, **kwargs):
            return SimpleNamespace(return_code=9, stdout="", stderr="scorer infrastructure failure")

    @asynccontextmanager
    async def owned(box, receipt):
        try:
            yield box
        finally:
            state["closed"] = True

    monkeypatch.setattr(verifier, "AsyncSandbox", lambda *args: Box())
    monkeypatch.setattr(verifier, "_owned_sandbox", owned)
    with pytest.raises(RuntimeError, match="not a scientific zero"):
        await verifier.regrade_managed(
            source=source,
            task_id="t_test--dfeep__paper_93__task_7",
            workspace=workspace,
            output=tmp_path / "output",
            capture_complete=True,
        )
    assert state["closed"]
    assert json.loads((tmp_path / "output/execution.json").read_text())["return_code"] == 9
    assert not (tmp_path / "output/score.json").exists()


def test_fifo_result_is_rejected_without_opening(tmp_path):
    import os

    os.mkfifo(tmp_path / "result.json")
    with pytest.raises(ValueError, match="non-regular"):
        validate_workspace("t_test--dfeep__paper_93__task_7", tmp_path, capture_complete=True)
