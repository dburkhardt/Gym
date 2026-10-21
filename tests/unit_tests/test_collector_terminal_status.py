# SPDX-License-Identifier: Apache-2.0
"""Terminal snapshots reflect durable rows, without inventing dispatch state."""

import asyncio
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import nemo_gym.rollout_collection as rc
from nemo_gym.batch_status import BatchStatusTracker
from tests.unit_tests.test_batch_status import materialized_row, write_manifest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["cancel", "cancel_status_error", "iterator_error", "spool_error", "close_error", "normal"]
)
async def test_terminal_snapshot_after_throttled_failures(tmp_path, monkeypatch, mode):
    rows = [materialized_row("agent", "test", i, 0) for i in range(3)]
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "results.jsonl"
    manifest = write_manifest(tmp_path / "batch_manifest.json", rows)
    config = rc.RolloutCollectionConfig(
        input_jsonl_fpath=str(inputs),
        output_jsonl_fpath=str(output),
        batch_manifest_fpath=str(manifest),
        disable_health_check=True,
        disable_aggregation=True,
        route_failures_to_sidecar=True,
    )
    if mode == "spool_error":
        config.retain_results_in_memory = False
        config.upload_rollouts = True
        monkeypatch.setattr(rc, "get_exporters", lambda: [object()])
        monkeypatch.setattr(rc, "export_metrics", lambda *a, **kw: None)
        original_unlink = Path.unlink

        def unlink(path, *a, **kw):
            if str(path).endswith(".upload.tmp"):
                raise OSError("fixture spool failure")
            return original_unlink(path, *a, **kw)

        monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(rc, "get_global_config_dict", lambda: OmegaConf.create({}))
    monkeypatch.setattr(rc.RolloutCollectionHelper, "_preprocess_rows_from_config", lambda *a: rows)
    blocked = asyncio.Event()
    settled = asyncio.Event()
    iterator_closed = asyncio.Event()
    finalizing = False
    original_write = BatchStatusTracker.write_status

    def write(self, completed, failed, **kwargs):
        self.write_interval_seconds = 3600
        if mode == "cancel_status_error" and finalizing and kwargs.get("force"):
            raise OSError("fixture publication failure")
        return original_write(self, completed, failed, **kwargs)

    monkeypatch.setattr(BatchStatusTracker, "write_status", write)

    async def worker(row):
        if row["_ng_task_index"] < 2:
            return rc._CompletedRollout(
                row, {"_ng_failure_class": "agent_run_error", "_ng_failure_message": "fixture"}, 0.0
            )
        blocked.set()
        try:
            if mode in ("normal", "close_error"):
                return rc._CompletedRollout(row, {"reward": 0.0}, 0.0)
            if mode in ("iterator_error", "spool_error"):
                raise RuntimeError("fixture iterator error")
            await asyncio.Event().wait()
        finally:
            settled.set()

    class Iterator:
        def __init__(self):
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index == len(rows):
                raise StopIteration
            row = rows[self.index]
            self.index += 1
            return worker(row)

        async def aclose(self):
            nonlocal finalizing
            finalizing = True
            iterator_closed.set()
            if mode == "close_error":
                raise OSError("fixture close failure")

    monkeypatch.setattr(rc.RolloutCollectionHelper, "_run_examples_with_metadata", lambda *a, **kw: Iterator())
    task = asyncio.create_task(rc.RolloutCollectionHelper().run_from_config(config))
    await asyncio.wait_for(blocked.wait(), 10)
    failure_path = rc.failures_path_for(output)
    before = failure_path.read_bytes()
    assert len(before.splitlines()) == 2
    status_path = manifest.parent / "batch_status.json"
    if mode.startswith("cancel"):
        status_before = json.loads(status_path.read_text())
        assert status_before["members"]["agent"]["failures_by_class"] == {"agent_run_error": 1}
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as error:
            await task
        if mode == "cancel_status_error":
            assert "publication failure" in str(error.value.__notes__)
    elif mode in ("iterator_error", "spool_error"):
        with pytest.raises(RuntimeError, match="fixture iterator error"):
            await task
    elif mode == "close_error":
        with pytest.raises(OSError, match="fixture close failure"):
            await task
    else:
        await task
    assert settled.is_set() and iterator_closed.is_set()
    assert failure_path.read_bytes() == before
    if mode != "cancel_status_error":
        status = json.loads(status_path.read_text())
        assert status["members"]["agent"]["failures_by_class"] == {"agent_run_error": 2}
    if mode not in ("normal", "close_error"):
        assert output.read_bytes() == b""
