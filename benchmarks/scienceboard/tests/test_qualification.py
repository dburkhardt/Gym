# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import json
import threading
from pathlib import Path

import pytest

from benchmarks.scienceboard import qualification as q


def test_receipt_requires_current_implementation_and_runtime():
    implementation = {"interpreter": "/venv/bin/python", "modules": {"worker": "sha"}}
    required = {"image_id": "sha256:image", "task_id": "task"}
    receipt = {
        **required,
        "implementation": implementation,
        "runtime": {
            "executable": "/venv/bin/python",
            "loaded_modules": {"actual.py": "sha"},
            "packages": [{"name": "native", "version": "1"}],
        },
        "hard_death_signal": "SIGKILL",
        "cleanup": {
            "all_removed": True,
            "containers": [{"all_removed": True, "ownership_labels": {"nemo-gym.run": "a" * 32}}],
        },
    }
    assert q.validate_receipt(receipt, implementation=implementation, required=required)
    for key, replacement in (("implementation", {}), ("runtime", {}), ("cleanup", {}), ("image_id", "changed")):
        changed = copy.deepcopy(receipt)
        changed[key] = replacement
        with pytest.raises(ValueError):
            q.validate_receipt(changed, implementation=implementation, required=required)


def test_cleanup_uses_precreate_intent_even_without_descriptor(tmp_path, monkeypatch):
    directory = tmp_path / "desktops"
    directory.mkdir()
    labels = {"nemo-gym.run": "a" * 32, "nemo-gym.workload": "scienceboard-desktop"}
    (directory / "owned.intent.json").write_text(json.dumps({"ownership_labels": labels}))
    observed = []

    class Provider:
        async def reconcile_owned(self, owner):
            observed.append(owner)
            return {"ownership_labels": owner, "all_removed": True, "removed_ids": ["immutable"]}

        async def aclose(self):
            pass

    monkeypatch.setattr(q, "DockerProvider", Provider)
    result = q.cleanup_desktops(directory)
    assert observed == [labels]
    assert result["containers"][0]["removed_ids"] == ["immutable"]


async def test_repeated_cancel_waits_for_cleanup(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def cleanup(path):
        entered.set()
        release.wait(5)
        return {"all_removed": True}

    monkeypatch.setattr(q, "cleanup_desktops", cleanup)
    job = asyncio.create_task(q.settle_cleanup(tmp_path))
    await asyncio.to_thread(entered.wait, 3)
    job.cancel()
    await asyncio.sleep(0.01)
    job.cancel()
    assert not job.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await job


async def test_missing_image_fails_before_output_or_worker(tmp_path):
    with pytest.raises(ValueError, match="not admitted"):
        await q.qualify({"image": {}, "audit": {}, "resources": {}, "output": str(tmp_path / "new")}, Path("/missing"))
    assert not (tmp_path / "new").exists()


def test_empty_precreate_lookup_remains_unresolved(tmp_path, monkeypatch):
    directory = tmp_path / "desktops"
    directory.mkdir()
    (directory / "run.intent.json").write_text(json.dumps({"ownership_labels": {"nemo-gym.run": "a" * 32}}))

    class Provider:
        async def reconcile_owned(self, labels):
            return {"ownership_labels": labels, "all_removed": True, "removed_ids": []}

        async def aclose(self):
            pass

    monkeypatch.setattr(q, "DockerProvider", Provider)
    with pytest.raises(RuntimeError, match="cleanup remains unresolved"):
        q.cleanup_desktops(directory)
    assert json.loads((tmp_path / "desktop-cleanup.json").read_text())["all_removed"] is False


async def test_worker_exception_during_settlement_still_cleans(tmp_path, monkeypatch):
    entered = asyncio.Event()
    observed = []

    async def worker():
        entered.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.sleep(0.01)
            raise RuntimeError("worker settlement failed")

    async def cleanup(path):
        observed.append(path)
        return {"all_removed": True}

    monkeypatch.setattr(q, "settle_cleanup", cleanup)
    job = asyncio.create_task(worker())
    await entered.wait()
    with pytest.raises(RuntimeError, match="worker settlement failed"):
        await q.stop_worker(job, tmp_path)
    assert observed == [tmp_path]


def test_runtime_fingerprint_reads_loaded_bytes_and_rejects_external_editable(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    module = ModuleType("scienceboard_qualification_fixture")
    file = tmp_path / "dependency.py"
    file.write_text("VALUE = 1\n")
    module.__file__ = str(file)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(q.importlib.metadata, "distributions", lambda: [])
    first = q.runtime_identity()["loaded_modules"][str(file)]
    file.write_text("VALUE = 2\n")
    assert q.runtime_identity()["loaded_modules"][str(file)] != first

    class Editable:
        metadata = {"Name": "external-scientific-package"}
        version = "1"

        def read_text(self, name):
            return json.dumps({"url": tmp_path.as_uri(), "dir_info": {"editable": True}})

    monkeypatch.setattr(q.importlib.metadata, "distributions", lambda: [Editable()])
    with pytest.raises(ValueError, match="External editable"):
        q.runtime_identity()
