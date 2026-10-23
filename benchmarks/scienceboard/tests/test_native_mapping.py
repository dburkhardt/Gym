# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from benchmarks.scienceboard.result_mapping import read_creator_result
from nemo_gym.result_disposition import summarize_dispositions
from responses_api_agents.scienceboard_native.app import RunRequest, map_result


def test_budget_null_keeps_native_identity_without_placeholder_score(tmp_path):
    body = RunRequest.model_validate(
        dict(
            task_id="frozen-1",
            verifier_metadata={},
            responses_create_params={"input": [{"role": "user", "content": "instruction"}]},
            _ng_task_index=3,
            _ng_rollout_index=2,
            _ng_attempt_id=1,
        )
    )
    record = read_creator_result(tmp_path / "missing", task_id="frozen-1", transport_outcome="model_output_budget")
    result = map_result(body, record, tmp_path).model_dump()
    assert result["_ng_task_index"] == 3
    assert result["_ng_rollout_index"] == 2
    assert result["_ng_attempt_id"] == 1
    assert result["metadata"]["scienceboard"]["attempted"] is True
    assert result["metadata"]["scienceboard"]["raw_reward"] is None
    assert result["failure_kind"] == "model_budget"
    summary = summarize_dispositions([result], expected_count=1)
    assert summary["counts"]["raw_scored_records"] == 0
    assert summary["counts"]["quality_selected_records"] == 0
    assert summary["counts"]["completed_result_records"] == 1


def test_real_zero_remains_scored(tmp_path):
    body = RunRequest.model_validate(dict(verifier_metadata={}, responses_create_params={"input": []}))
    path = tmp_path / "result.out"
    path.write_text("0")
    record = read_creator_result(path, task_id="frozen-1", transport_outcome="captured_model_action")
    result = map_result(body, record, tmp_path).model_dump()
    assert result["reward"] == 0 and result["mask_sample"] is False
    assert summarize_dispositions([result])["counts"]["raw_scored_records"] == 1


async def test_native_run_refuses_unqualified_runtime_before_provider(tmp_path):
    import json
    from types import SimpleNamespace

    import pytest

    from benchmarks.scienceboard.prepare import MANIFEST
    from benchmarks.scienceboard.tests.test_admission import evidence
    from benchmarks.scienceboard.tests.test_transport import SOURCE
    from responses_api_agents.scienceboard_native.app import Agent

    if not SOURCE.is_dir():
        pytest.skip("Pinned creator source required; set SCIENCEBOARD_SOURCE_ROOT")
    item = json.loads(MANIFEST.read_text())["tasks"][0]
    task = json.loads((SOURCE / item["path"]).read_text())
    manifest = json.loads(MANIFEST.read_text())
    paths = []
    for name, value in zip(("inspect", "audit", "placement"), evidence()):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        paths.append(path)
    cfg = SimpleNamespace(
        creator_source=SOURCE,
        image_inspect=paths[0],
        image_audit=paths[1],
        placement=paths[2],
        runtime_qualified=False,
    )
    body = RunRequest.model_validate(
        dict(
            verifier_metadata=dict(
                task_id=item["task_id"],
                task_sha256=item["sha256"],
                creator_task_path=item["path"],
                source_revision=manifest["source_revision"],
            ),
            responses_create_params={"input": [{"role": "user", "content": task["instruction"]}]},
        )
    )
    with pytest.raises(ValueError, match="runtime unqualified"):
        await Agent._run(SimpleNamespace(config=cfg), body)
    assert sorted(x.name for x in tmp_path.iterdir()) == ["audit", "inspect", "placement"]
    body.verifier_metadata["task_sha256"] = "changed"
    with pytest.raises(ValueError, match="Frozen task metadata"):
        await Agent._run(SimpleNamespace(config=cfg), body)


def test_actual_tester_swallowed_exception_attribution(tmp_path, monkeypatch):
    """Real Tester catches TaskInfo exceptions; wrapper must retain their origin."""
    import importlib
    import json
    from types import SimpleNamespace

    import pytest

    from benchmarks.scienceboard import worker
    from benchmarks.scienceboard.guarded_model import CapturedModelActionError
    from benchmarks.scienceboard.prepare import MANIFEST
    from benchmarks.scienceboard.tests.test_transport import SOURCE

    tester_module = pytest.importorskip("sci.Tester")
    manager_module = importlib.import_module("sci.base.manager")
    monkeypatch.setattr(manager_module.Manager, "pause", staticmethod(lambda *a: None))
    # No GUI allocation: exercise actual Tester logging/catching and our worker binding.
    monkeypatch.setattr(tester_module.TaskGroup, "__call__", lambda self, *a: iter(sum(self.groups, [])))
    monkeypatch.setattr(worker, "creator_desktop_factory", lambda *a, **kw: object)
    # Desktop controller package is unavailable locally. Substitute only its
    # unused base class; actual creator Tester/TaskGroup/TaskInfo are imported.
    import sys
    from types import ModuleType

    controller = ModuleType("desktop_env.controllers.python")
    controller.PythonController = object
    monkeypatch.setitem(sys.modules, "desktop_env.controllers.python", controller)
    original_desktop = importlib.import_module("sci.docker").DockerDesktopEnv
    item = json.loads(MANIFEST.read_text())["tasks"][0]
    monkeypatch.setenv("SCIENCEBOARD_BRIDGE_TOKEN", "fixture")
    import hashlib

    modules = [
        worker,
        *[
            importlib.import_module(name)
            for name in (
                "benchmarks.scienceboard.guarded_model",
                "benchmarks.scienceboard.managed_desktop",
                "benchmarks.scienceboard.prepare",
                "benchmarks.scienceboard.result_mapping",
                "nemo_gym.sandbox.guarded_chat_transport",
                "nemo_gym.guarded_chat_transport",
                "benchmarks.scienceboard.qualification",
            )
        ],
    ]
    implementation = {
        str(Path(m.__file__).resolve()): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules
    }
    monkeypatch.setattr(worker, "runtime_identity", lambda: {"fixture": "same native dependency state"})
    for kind in ("model", "creator"):
        out = tmp_path / kind
        out.mkdir()

        def fail(info):
            if kind == "model":
                raise CapturedModelActionError("captured length")
            raise RuntimeError("independent creator failure")

        monkeypatch.setattr(tester_module.TaskInfo, "__call__", fail)
        monkeypatch.setattr(
            worker, "CreatorModelHTTP", lambda **kw: SimpleNamespace(outcome="model_output_budget", close=lambda: None)
        )
        record = worker.run(
            dict(
                implementation=implementation,
                expected_runtime=worker.runtime_identity(),
                source=str(SOURCE),
                output=str(out),
                task_id=item["task_id"],
                image={"Id": "sha256:" + "a" * 64},
                audit={},
                resources={},
                socket="unused",
                lineage="fixture",
                digest="a" * 64,
                protocol="fixture",
            )
        )
        assert record["raw_reward"] is None
        assert record["mask_sample"] is (kind == "creator")
        assert bool(record["failures"]) is (kind == "creator")
        assert importlib.import_module("sci.docker").DockerDesktopEnv is original_desktop
        assert tester_module.TaskInfo.__call__ is fail
