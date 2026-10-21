# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from harbor.models.trial.config import EnvironmentConfig

from responses_api_agents.harbor_agent.app import run_harbor_job
from responses_api_agents.harbor_agent.tests.test_app import _make_server
from responses_api_agents.harbor_agent.utils import HarborAgentUtils


def _task(tmp_path, resources):
    (tmp_path / "environment").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test.sh").write_text("#!/bin/sh\ntrue\n")
    (tmp_path / "instruction.md").write_text("Deterministic control")
    (tmp_path / "task.toml").write_text('[environment]\ndocker_image="example/control"\n' + resources)
    return tmp_path


def test_explicit_resources_preserve_scientific_values(tmp_path):
    task = _task(tmp_path, "cpus=8\nmemory_mb=16384\n")
    before = (task / "task.toml").read_bytes()
    HarborAgentUtils.validate_task_resources(task, EnvironmentConfig())
    assert (task / "task.toml").read_bytes() == before


def test_omitted_resources_fail_before_execution(tmp_path):
    task = _task(tmp_path, "")
    with pytest.raises(ValueError, match="explicit environment cpus, memory_mb"):
        HarborAgentUtils.validate_task_resources(task, EnvironmentConfig())


def test_native_validity_is_derived_without_mutating_raw_receipt():
    raw = json.loads((Path(__file__).parent / "fixtures" / "harbor_96ff9c6_separate_control.json").read_text())
    assert "reward_valid" not in raw
    derived = HarborAgentUtils.with_native_validity(raw)
    assert derived["reward_valid"] is True
    assert "reward_valid" not in raw
    raw["exception_info"] = {
        "exception_type": "VerifierTimeoutError",
        "exception_message": "injected",
        "exception_traceback": "control",
        "occurred_at": "2026-10-20T00:00:00Z",
    }
    assert HarborAgentUtils.with_native_validity(raw)["reward_valid"] is False


def test_job_config_keeps_one_trial_and_explicit_zero_retry(tmp_path):
    config = _make_server()._build_job_config(
        "scientific", "task", "test_model", "http://localhost:1/v1", "job", tmp_path
    )
    assert config["n_concurrent_trials"] == 1
    assert config["n_attempts"] == 1
    assert config["retry"]["max_retries"] == 0
    assert config["datasets"][0]["path"] == "/tmp/test_dataset"


@pytest.mark.asyncio
async def test_runner_awaits_native_create_and_retains_result_on_run_failure(tmp_path):
    trial = tmp_path / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text("{}")
    job = SimpleNamespace(_task_download_results={}, run=AsyncMock(side_effect=RuntimeError("recorded failure")))
    with patch("harbor.job.Job.create", new=AsyncMock(return_value=job)) as create:
        result = await run_harbor_job({"job_name": "job", "jobs_dir": str(tmp_path)})
    create.assert_awaited_once()
    job.run.assert_awaited_once()
    assert result == str(trial.resolve())


@pytest.mark.asyncio
async def test_native_failure_propagates_after_diagnostics(tmp_path):
    from harbor.agents.terminus_2.terminus_2 import Terminus2

    from responses_api_agents.harbor_agent.custom_agents.airs_no_retry import AIRSNoRetryAgent

    agent = AIRSNoRetryAgent(
        logs_dir=tmp_path,
        model_name="nvidia/nvidia/nemotron-3.5-super-vl",
        api_base="http://127.0.0.1:1/v1",
        model_info={"max_input_tokens": 49152, "max_output_tokens": 16384},
        enable_summarize=False,
    )
    with (
        patch.object(Terminus2, "run", new=AsyncMock(side_effect=ConnectionError("provider unavailable"))),
        patch.object(agent._llm, "aclose", new=AsyncMock(side_effect=RuntimeError("cleanup failure"))) as close,
    ):
        with pytest.raises(ConnectionError, match="provider unavailable"):
            await agent.run("control", None, None)
    close.assert_awaited_once()
    assert (tmp_path / "agent_error_flags.json").exists()


@pytest.mark.parametrize("exception_type, expected", [("AgentTimeoutError", True), ("APIConnectionError", False)])
def test_native_budget_and_transport_policy_remain_distinct(exception_type, expected):
    raw = json.loads((Path(__file__).parent / "fixtures" / "harbor_96ff9c6_separate_control.json").read_text())
    raw["exception_info"] = {
        "exception_type": exception_type,
        "exception_message": "control",
        "exception_traceback": "control",
        "occurred_at": "2026-10-20T00:00:00Z",
    }
    assert HarborAgentUtils.with_native_validity(raw)["reward_valid"] is expected


def test_native_terminus_preflight_accepts_gym_model_options():
    from responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym import Terminus2NemoGym

    parsed = Terminus2NemoGym.parse_options(
        {
            "parser_name": "json",
            "responses_create_params": {"max_output_tokens": 16384},
            "nemo_model_server_timeout_sec": 120.0,
        }
    )
    assert parsed.responses_create_params == {"max_output_tokens": 16384}
    assert parsed.parser_name == "json"


def test_production_terminus_job_preflight_and_guard(tmp_path):
    from harbor.job import Job
    from harbor.models.job.config import JobConfig

    from responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym import Terminus2NemoGym

    kwargs = {
        "interleaved_thinking": True,
        "enable_summarize": False,
        "collect_rollout_details": True,
        "trajectory_config": {"raw_content": True},
        "model_info": {
            "max_input_tokens": 49152,
            "max_output_tokens": 16384,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
        "max_turns": 40,
        "parser_name": "json",
        "responses_create_params": {"input": [], "temperature": 0.0, "max_output_tokens": 16384},
    }
    server = _make_server(
        harbor_agent_import_path="responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym:Terminus2NemoGym",
        harbor_agent_kwargs=kwargs,
    )
    raw = server._build_job_config(
        dataset_alias="scientific",
        task_name="test_task_123",
        model_name="authorized-model",
        api_base="http://localhost:9999/ng-rollout/0-0/v1",
        job_name="control",
        jobs_dir=tmp_path,
        gym_provider_dispatch_guard=True,
    )
    config = JobConfig.model_validate(raw)
    Job._preflight_agents(config, config.agents)
    options = config.agents[0].kwargs
    assert options["gym_provider_dispatch_guard"] is True
    agent = Terminus2NemoGym(logs_dir=tmp_path, model_name="authorized-model", **options)
    assert agent._nemo_gym_llm._dispatch_path == tmp_path / "gym-call-state.json"
    assert agent._nemo_gym_llm._dispatch_state["calls"] == {}
    with pytest.raises(ValueError, match="rollout-bound"):
        Terminus2NemoGym(
            logs_dir=tmp_path, model_name="authorized-model", **{**options, "api_base": "http://localhost:9999/v1"}
        )
