# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest


pytest.importorskip("verifiers.v1", reason="Pinned optional creator runtime is not installed")

from benchmarks.bioagentbench import creator_execution as execution


@pytest.mark.asyncio
@pytest.mark.parametrize("task_id", ["cystic-fibrosis", "giab"])
async def test_native_environment_configuration_and_restoration(tmp_path, monkeypatch, task_id):
    import bioagent_taskset.taskset as creator
    import verifiers.v1.runtimes as runtimes
    from bioagent_opencode.harness import OpenCodeHarness
    from verifiers.v1 import judge
    from verifiers.v1.cli.eval import runner

    source = tmp_path / "assets" / task_id
    (source / "data").mkdir(parents=True)
    metadata = tmp_path / "metadata.json"
    metadata.write_text("[]")
    original = (runtimes.DockerRuntime, judge.build_async_openai, OpenCodeHarness.launch)
    observed = []
    raw_metrics = {"status": "hap.py_failed", "f1_score": 0.0, "exit_code": 255}

    async def scorer(*args):
        return raw_metrics

    monkeypatch.setattr(creator, "benchmark_metrics", scorer)

    async def run(env, config):
        observed.append(config.model_dump(mode="json"))
        assert runtimes.DockerRuntime is not original[0]
        client = judge.build_async_openai(config.taskset.task.judge)
        assert client.max_retries == 0
        await client.close()
        if task_id == "giab":
            from types import SimpleNamespace

            assert creator.benchmark_metrics is not scorer
            assert await creator.benchmark_metrics("giab", source, source) is raw_metrics
            return [SimpleNamespace(model_dump=lambda **kwargs: {"info": {}, "rewards": {"results_match": 0.0}})]
        assert creator.benchmark_metrics is scorer
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
            "task_id": task_id,
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
        "task_id": task_id,
        "image": "sha256:" + "a" * 64,
        "endpoint": "http://local/ng-rollout/one/v1",
        "lineage": "one",
        "digest": "digest",
        "model": "model",
        "model_socket": str(tmp_path / "model.sock"),
        "runtime_socket": str(tmp_path / "runtime.sock"),
        "ipc_token": "fixture",
        "deadline": 9999999999,
        "runtime_config": {},
    }
    (tmp_path / "run").mkdir()
    if task_id == "giab":
        result = await execution.run_native(config=config, packet=packet, output=tmp_path / "run")
        assert result[0]["info"]["gym_giab_benchmark_metrics"] == raw_metrics
        assert result[0]["rewards"]["results_match"] == 0
    else:
        with pytest.raises(RuntimeError, match="controlled native"):
            await execution.run_native(config=config, packet=packet, output=tmp_path / "run")
    assert creator.benchmark_metrics is scorer
    assert (runtimes.DockerRuntime, judge.build_async_openai, OpenCodeHarness.launch) == original
    assert observed and not (tmp_path / "run/calls").exists()


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


@pytest.mark.asyncio
async def test_relay_readiness_preserves_definite_exit_and_reads_once():
    from types import SimpleNamespace

    calls = []

    class Runtime:
        async def run(self, argv, env):
            assert argv == ["test", "-s", ".vf-relay-port.json"] and env == {}
            calls.append("test")
            return SimpleNamespace(exit_code=1 if len(calls) == 1 else 0)

        async def read(self, path):
            calls.append("read")
            return b'{"port": 12345}'

    assert await execution.relay_port(Runtime()) == 12345
    assert calls == ["test", "test", "read"]


@pytest.mark.asyncio
async def test_relay_unknown_read_is_never_retried():
    import asyncio
    from types import SimpleNamespace

    calls = []

    class Runtime:
        async def run(self, argv, env):
            calls.append("test")
            return SimpleNamespace(exit_code=0)

        async def read(self, path):
            calls.append("read")
            raise asyncio.IncompleteReadError(b"", 4)

    with pytest.raises(asyncio.IncompleteReadError):
        await execution.relay_port(Runtime())
    assert calls == ["test", "read"]
