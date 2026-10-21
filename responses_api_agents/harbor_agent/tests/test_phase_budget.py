# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from harbor.models.task.task import Task
from harbor.models.trial.config import TrialConfig

from responses_api_agents.harbor_agent.app import run_harbor_job
from responses_api_agents.harbor_agent.phase_budget import resolve_phase_budget, validate_admission_budget
from responses_api_agents.harbor_agent.tests.test_upgrade_contract import _task


def configured(tmp_path, mode="shared", **overrides):
    path = _task(tmp_path, "cpus=1\nmemory_mb=4096\nbuild_timeout_sec=600\n")
    with (path / "task.toml").open("a") as stream:
        stream.write(f'[agent]\ntimeout_sec=1800\n[verifier]\ntimeout_sec=60\nenvironment_mode="{mode}"\n')
    config = TrialConfig.model_validate(
        {
            "task": {"path": str(path)},
            "agent": {
                "import_path": "responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym:Terminus2NemoGym"
            },
            **overrides,
        }
    )
    return config, Task(path)


@pytest.mark.parametrize("mode,expected", [("shared", 2820), ("separate", 3420)])
def test_pinned_properties_include_setup_and_separate_start(tmp_path, mode, expected):
    config, task = configured(tmp_path, mode)
    before = task.paths.config_path.read_bytes()
    report = resolve_phase_budget(config, task)
    assert report["declared_phase_sum_seconds"] == expected
    assert report["phases_seconds"]["agent_setup"] == 360
    assert ("verifier_environment_start" in report["phases_seconds"]) == (mode == "separate")
    assert task.paths.config_path.read_bytes() == before
    assert not (tmp_path / "trials").exists()
    assert not report["end_to_end_runtime_bound_proven"]
    assert report["sources"]["trial"]["sha256"]
    with pytest.raises(ValueError, match="below declared"):
        validate_admission_budget(report, drain_seconds=expected, operational_allowance_seconds=70)
    assert validate_admission_budget(report, drain_seconds=expected + 70, operational_allowance_seconds=70)


def test_override_max_and_phase_multipliers_use_upstream_resolution(tmp_path):
    config, task = configured(
        tmp_path,
        "separate",
        timeout_multiplier=2,
        agent_setup_timeout_multiplier=3,
        environment_build_timeout_multiplier=4,
    )
    config.agent.override_timeout_sec = 1000
    config.agent.max_timeout_sec = 800
    config.agent.override_setup_timeout_sec = 10
    config.verifier.override_timeout_sec = 90
    config.verifier.max_timeout_sec = 70
    assert resolve_phase_budget(config, task)["phases_seconds"] == {
        "agent_environment_start": 2400,
        "agent_setup": 30,
        "agent_execution": 1600,
        "verifier_environment_start": 2400,
        "verifier_execution": 140,
    }


def test_disabled_verifier_has_no_second_start(tmp_path):
    config, task = configured(tmp_path, "separate")
    config.verifier.disable = True
    assert resolve_phase_budget(config, task)["declared_phase_sum_seconds"] == 2760


@pytest.mark.parametrize("kind", ["steps", "user", "regrade", "install", "trajectory"])
def test_unsupported_lifecycles_fail_closed(tmp_path, kind):
    config, task = configured(tmp_path)
    if kind == "steps":
        task = SimpleNamespace(has_steps=True)
    elif kind == "user":
        config.user_agent = object()
    elif kind == "regrade":
        config.source_trial = object()
    elif kind == "install":
        config.install_only = True
    else:
        config.agent.load_trajectory = "previous.json"
    with pytest.raises(ValueError, match="supports only|does not support"):
        resolve_phase_budget(config, task)


@pytest.mark.parametrize("value", [None, 0, -1, True, float("inf"), float("nan")])
def test_invalid_allowance_refused(value):
    with pytest.raises(ValueError, match="finite positive"):
        validate_admission_budget(
            {"declared_phase_sum_seconds": 1}, drain_seconds=10, operational_allowance_seconds=value
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2])
async def test_actual_worker_rejects_before_job_run_preserving_receipt(tmp_path, count):
    config, task = configured(tmp_path, "separate")
    job_config = SimpleNamespace(jobs_dir=tmp_path, job_name="job")
    (tmp_path / "job").mkdir()
    job = SimpleNamespace(
        config=job_config,
        _trial_configs=[config] * count,
        _task_download_results={config.task.get_task_id(): SimpleNamespace(path=tmp_path)},
        run=AsyncMock(),
    )
    with patch("harbor.job.Job.create", new=AsyncMock(return_value=job)):
        with pytest.raises(ValueError, match="below declared|exactly one"):
            await run_harbor_job(
                {"job_name": "job", "jobs_dir": str(tmp_path)},
                {"drain_seconds": 100, "operational_allowance_seconds": 70},
            )
    job.run.assert_not_awaited()
    if count == 1:
        receipt = json.loads((tmp_path / "job/gym-phase-budget.json").read_text())
        assert receipt["trials"][0]["declared_phase_sum_seconds"] == 3420
    assert not list((tmp_path / "job").glob("*/result.json"))


@pytest.mark.asyncio
async def test_parent_binds_actual_collector_drain(monkeypatch):
    from responses_api_agents.harbor_agent.tests.test_app import (
        _GLOBAL_CONFIG,
        _harbor_run_mocks,
        _make_run_request,
        _make_server,
    )

    monkeypatch.setitem(_GLOBAL_CONFIG, "drain_margin_s", 3500)
    with _harbor_run_mocks() as runner:
        await _make_server(harbor_admission_finalization_allowance_s=70).run(_make_run_request())
        params = runner.remote.call_args.args[1]
        assert params["admission_budget"] == {"drain_seconds": 3500, "operational_allowance_seconds": 70}


@pytest.mark.asyncio
async def test_sufficient_worker_budget_preserves_native_success(tmp_path):
    config, task = configured(tmp_path, "separate")
    trial = tmp_path / "job" / "trial"
    trial.mkdir(parents=True)
    native = b'{"scientific": "preserved", "reward": 0.0}\n'
    (trial / "result.json").write_bytes(native)
    job = SimpleNamespace(
        config=SimpleNamespace(jobs_dir=tmp_path, job_name="job"),
        _trial_configs=[config],
        _task_download_results={config.task.get_task_id(): SimpleNamespace(path=tmp_path)},
        run=AsyncMock(),
    )
    with patch("harbor.job.Job.create", new=AsyncMock(return_value=job)):
        result = await run_harbor_job(
            {"job_name": "job", "jobs_dir": str(tmp_path)},
            {"drain_seconds": 3490, "operational_allowance_seconds": 70},
        )
    job.run.assert_awaited_once()
    assert result == str(trial.resolve())
    assert (trial / "result.json").read_bytes() == native
