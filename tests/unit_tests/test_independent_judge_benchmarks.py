# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.config_types import ConfigError
from nemo_gym.eval_compatibility import validate_eval_compatibility
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


ROOT = Path(__file__).resolve().parents[2]
MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


@pytest.fixture(params=["ctbench", "fdarxbench"])
def configured(request, tmp_path, monkeypatch):
    name = request.param
    rubric = tmp_path / "rubric.txt"
    rubric.write_text("Planning fixture only; never submitted to a judge.")
    monkeypatch.setenv("FDARX_JUDGE_PROMPT", str(rubric))
    row = {
        "task_id": "fixture-1",
        "responses_create_params": {"input": [{"role": "user", "content": "planning fixture"}]},
        "verifier_metadata": {"answer": "fixture"},
    }
    sample = tmp_path / "sample.jsonl"
    sample.write_text(json.dumps(row) + "\n")
    dataset = OmegaConf.to_container(
        OmegaConf.load(ROOT / f"benchmarks/{name}/config.yaml")[name].resources_servers[name].datasets[0]
    )
    dataset["jsonl_fpath"] = str(sample)
    value = {
        "config_paths": [f"benchmarks/{name}/config.yaml"],
        name: {"resources_servers": {name: {"datasets": [dataset]}}},
        "use_cached_prepared_benchmarks": True,
        "judge_base_url": "http://fixture.invalid/v1",
        "judge_api_key": "not-a-credential",
        "policy_model": {
            "responses_api_models": {
                "openai_model": {
                    "entrypoint": "app.py",
                    "openai_model": "independent-solver",
                    "openai_base_url": "http://unset.local/v1",
                    "openai_api_key": "unset",
                }
            }
        },
    }
    return name, value, sample


def resolve(value):
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(value),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )


def test_actual_discovery_and_independent_model_binding(configured):
    name, value, _ = configured
    assert BenchmarkConfig.from_config_path(ROOT / f"benchmarks/{name}/config.yaml", strict=False).name == name
    config = resolve(value)
    assert config.judge_model.responses_api_models.openai_model.openai_model == MODEL
    value["policy_model"]["responses_api_models"]["openai_model"]["openai_model"] = "changed-solver"
    assert resolve(value).judge_model.responses_api_models.openai_model.openai_model == MODEL
    value["judge_model_name"] = "explicit-judge-variant"
    assert resolve(value).judge_model.responses_api_models.openai_model.openai_model == "explicit-judge-variant"


@pytest.mark.parametrize("capability", [None, False, True])
def test_judge_capability_admission(configured, capability):
    _, value, _ = configured
    value["component_capabilities"] = {"judge_model": {}}
    if capability is not None:
        value["component_capabilities"]["judge_model"]["text_generation"] = capability
    config = resolve(value)
    if capability is True:
        validate_eval_compatibility(config)
    else:
        with pytest.raises(ConfigError, match="unsupported" if capability is False else "undeclared"):
            validate_eval_compatibility(config)


def test_supported_cli_plan_without_services_or_private_assets(configured, tmp_path):
    name, value, sample = configured
    value["component_capabilities"] = {"judge_model": {"text_generation": True}}
    runtime = tmp_path / "runtime.yaml"
    OmegaConf.save(OmegaConf.create(value), runtime)
    output = tmp_path / "plan.json"
    before = sample.read_bytes()
    command = [
        sys.executable,
        "-c",
        "from nemo_gym.cli.main import main;main()",
        "eval",
        "plan",
        "--config",
        str(runtime),
        "--out",
        str(output),
        "--validate-compatibility",
    ]
    run = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT), "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert len(json.loads(output.read_text())["tasks"]) == 1
    assert sample.read_bytes() == before


@pytest.mark.parametrize("missing", ["judge_base_url", "judge_api_key"])
def test_actual_run_missing_judge_connection_refuses_before_preparation(configured, tmp_path, missing):
    _, value, sample = configured
    del value[missing]
    # A missing sample would fail preparation if runtime validation did not fail first.
    sample.unlink()
    runtime = tmp_path / "missing.yaml"
    OmegaConf.save(OmegaConf.create(value), runtime)
    output = tmp_path / "results.jsonl"
    command = [
        sys.executable,
        "-c",
        "from nemo_gym.cli.main import main;main()",
        "eval",
        "run",
        "--config",
        str(runtime),
        "--model-type",
        "openai_model",
        "--split",
        "benchmark",
        "--output",
        str(output),
    ]
    run = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT), "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode != 0
    assert missing in run.stdout + run.stderr
    assert "which is not set after merging" in run.stdout + run.stderr
    assert not output.exists() and not output.with_suffix("").exists() and not sample.exists()
