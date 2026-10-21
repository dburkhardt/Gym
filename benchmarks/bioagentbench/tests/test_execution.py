# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest


pytest.importorskip("verifiers.v1", reason="Pinned optional creator runtime is not installed")

from benchmarks.bioagentbench import execution


@pytest.mark.asyncio
async def test_native_environment_configuration_and_restoration(tmp_path, monkeypatch):
    import verifiers.v1.runtimes as runtimes
    from bioagent_opencode.harness import OpenCodeHarness
    from verifiers.v1 import judge
    from verifiers.v1.cli.eval import runner

    source = tmp_path / "assets" / "cystic-fibrosis"
    (source / "data").mkdir(parents=True)
    metadata = tmp_path / "metadata.json"
    metadata.write_text("[]")
    original = (runtimes.DockerRuntime, judge.build_async_openai, OpenCodeHarness.launch)
    observed = []

    async def run(env, config):
        observed.append(config.model_dump(mode="json"))
        assert runtimes.DockerRuntime is not original[0]
        client = judge.build_async_openai(config.taskset.task.judge)
        assert client.max_retries == 0
        await client.close()
        raise RuntimeError("controlled native pre-dispatch stop")

    monkeypatch.setattr(runner, "run_eval", run)
    config = {
        "model": "model",
        "num_rollouts": 1,
        "max_concurrent": 1,
        "taskset": {
            "id": "bioagent-taskset",
            "metadata_path": str(metadata),
            "data_root": str(source.parent),
            "task_id": "cystic-fibrosis",
            "staging": "local",
            "include_reference": True,
            "task": {"judge": {"model": "model"}},
        },
        "harness": {
            "id": "bioagent-opencode",
            "version": "1.15.11",
            "runtime": {"type": "docker", "image": "sha256:" + "a" * 64, "cpu": 2, "memory": 8},
        },
    }
    packet = {
        "task_id": "cystic-fibrosis",
        "image": "sha256:" + "a" * 64,
        "endpoint": "http://local/ng-rollout/one/v1",
        "lineage": "one",
        "digest": "digest",
        "model": "model",
    }
    with pytest.raises(RuntimeError, match="controlled native"):
        await execution.run_native(config=config, packet=packet, output=tmp_path / "run")
    assert (runtimes.DockerRuntime, judge.build_async_openai, OpenCodeHarness.launch) == original
    assert observed and list((tmp_path / "run/calls").iterdir()) == []


def test_all_ten_prepared_prompts_match_actual_creator():
    from pathlib import Path

    from bioagent_taskset.taskset import BioAgentConfig, BioAgentTaskset

    from benchmarks.bioagentbench.prepare import TASKS, rows

    workspace = Path(__file__).resolve().parents[5]
    metadata = (
        workspace
        / "outputs/broad-eval-campaign/harbor/agent-expansion/sources/56/bioagent-bench-bioagent-bench-6d098b6/src/task_metadata.json"
    )
    if not metadata.exists():
        pytest.skip("Pinned creator metadata fixture unavailable")
    actual = BioAgentTaskset(
        BioAgentConfig(metadata_path=metadata, data_root=Path("/nonserving/assets"), staging="local")
    ).load()
    prepared = rows(metadata)
    assert [row["task_id"] for row in prepared] == TASKS
    assert {task.data.task_id: task.data.prompt for task in actual} == {
        row["task_id"]: row["responses_create_params"]["input"][0]["content"] for row in prepared
    }


def test_actual_source_and_asset_identity_before_dispatch(tmp_path):
    import hashlib
    import json
    import sys
    from pathlib import Path

    from benchmarks.bioagentbench import worker

    root = Path(__file__).resolve().parents[3]
    workspace = root.parent.parent
    metadata = (
        workspace
        / "outputs/broad-eval-campaign/harbor/agent-expansion/sources/56/bioagent-bench-bioagent-bench-6d098b6/src/task_metadata.json"
    )
    creator = root.parent / "bioagent-experiments-upstream"
    if not metadata.exists() or not creator.exists():
        pytest.skip("Pinned native source fixture unavailable")
    asset = tmp_path / "assets/cystic-fibrosis/data"
    asset.mkdir(parents=True)
    (asset / "input").write_text("public input")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "tasks": {
                    "cystic-fibrosis": {
                        "files": {"data/input": hashlib.sha256((asset / "input").read_bytes()).hexdigest()}
                    }
                }
            }
        )
    )
    files = [
        *list((root / "benchmarks/bioagentbench").glob("*.py")),
        *list((root / "nemo_gym/sandbox").rglob("*.py")),
        root / "nemo_gym/server_utils.py",
    ]
    packet = {
        "creator_root": str(creator),
        "selected_sources": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        "interpreter": {
            "path": os.path.abspath(sys.executable),
            "sha256": hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
        },
        "asset_manifest": str(manifest),
        "asset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "task_id": "cystic-fibrosis",
    }
    config = {"taskset": {"metadata_path": str(metadata), "data_root": str(asset.parent.parent)}}
    identity = worker.validate_identity(packet, config)
    import subprocess

    packet["native_config"] = config
    packet["digest"] = "offline-control"
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(str(Path(p).absolute()) for p in sys.path if p)
    child = subprocess.run(
        [sys.executable, "-m", "benchmarks.bioagentbench.worker", "--packet", str(packet_path), "--validate-only"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr
    child_identity = json.loads((tmp_path / "actual-imports.json").read_text())
    assert child_identity["prefix"] == sys.prefix
    assert Path(child_identity["imports"]["bioagent_taskset.taskset"]["path"]).is_relative_to(
        tmp_path / "creator-source"
    )
    assert not (tmp_path / "creator-source/.pytest_cache").exists()
    assert (
        child_identity["imports"]["bioagent_worker"]["sha256"]
        == hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest()
    )
    assert not (tmp_path / "execution").exists()
    assert Path(identity["imports"]["bioagent_taskset.taskset"]["path"]).resolve().is_relative_to(creator.resolve())
    (asset / "input").write_text("changed")
    with pytest.raises(ValueError, match="asset packet differs"):
        worker.validate_identity(packet, config)


@pytest.mark.asyncio
async def test_actual_creator_judge_failure_does_not_publish_default_reward(tmp_path, monkeypatch):
    from pathlib import Path

    import bioagent_taskset.taskset as creator
    import verifiers.v1 as vf
    from verifiers.v1.errors import TaskError

    from benchmarks.bioagentbench.outcomes import disposition

    workspace = Path(__file__).resolve().parents[5]
    metadata = (
        workspace
        / "outputs/broad-eval-campaign/harbor/agent-expansion/sources/56/bioagent-bench-bioagent-bench-6d098b6/src/task_metadata.json"
    )
    task = creator.BioAgentTaskset(
        creator.BioAgentConfig(metadata_path=metadata, data_root=tmp_path, staging="local")
    ).load()[0]
    trace = vf.Trace(task={"type": type(task).__name__, "data": task.data})
    trace.info["artifacts"] = {"directory": str(tmp_path), "files": []}

    async def benchmark(*args):
        return {}

    async def score(*args):
        return False

    async def snippets(*args):
        return {}

    async def failed_judge(*args, **kwargs):
        raise RuntimeError("controlled judge failure")

    monkeypatch.setattr(creator, "benchmark_metrics", benchmark)
    monkeypatch.setattr(creator, "score_results", score)
    monkeypatch.setattr(creator, "runtime_table_snippets", snippets)
    monkeypatch.setattr(creator, "local_table_snippets", lambda *args: {})
    monkeypatch.setattr(creator.PipelineJudge, "evaluate", failed_judge)
    with pytest.raises(TaskError, match="controlled judge failure") as caught:
        await task.score(trace, object())
    assert trace.rewards == {}
    trace.capture_error(caught.value)
    trace.is_completed = True
    assert not disposition(trace.model_dump())["measurement_available"]
    # A parallel scoring branch could have assigned a reward before this failure.
    trace.rewards["results_match"] = 0.0
    assert not disposition(trace.model_dump())["measurement_available"]
