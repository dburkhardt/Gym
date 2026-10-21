# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "benchmark,capability", [("microvqa", "vision"), ("biomaze", "text_generation"), ("microvqa", "single_turn")]
)
@pytest.mark.parametrize("declared", [True, False, None])
def test_actual_cli_cells(tmp_path, benchmark, capability, declared):
    sample = ROOT / f"resources_servers/{benchmark}/data/benchmark.jsonl"
    if not sample.exists():
        pytest.skip("Prepared frozen sample required")
    before = hashlib.sha256(sample.read_bytes()).hexdigest()
    config = OmegaConf.merge(
        OmegaConf.load(ROOT / f"resources_servers/{benchmark}/configs/{benchmark}.yaml"),
        OmegaConf.load(ROOT / f"benchmarks/{benchmark}/config.yaml"),
    )
    del config.config_paths
    config.use_cached_prepared_benchmarks = True
    config.policy_model = {
        "responses_api_models": {
            "openai_model": {
                "entrypoint": "app.py",
                "openai_model": "declaration-only-model",
                "openai_base_url": "https://example.invalid/v1",
                "openai_api_key": "not-a-credential",
            }
        }
    }
    config.component_capabilities = {"policy_model": {}}
    dataset = config[benchmark].resources_servers[benchmark].datasets[0]
    if benchmark == "biomaze":
        config.judge_model = {
            "responses_api_models": {
                "openai_model": {
                    "entrypoint": "app.py",
                    "openai_model": "declaration-only-judge",
                    "openai_base_url": "https://example.invalid/v1",
                    "openai_api_key": "not-a-credential",
                }
            }
        }
        config.biomaze.resources_servers.biomaze.judge_model_server = {
            "type": "responses_api_models",
            "name": "judge_model",
        }
        dataset.required_model_roles = {
            "conclusion_judge": {
                "component": "biomaze",
                "binding_field": "judge_model_server",
                "capabilities": [capability],
            }
        }
        config.component_capabilities.judge_model = {}
        target = "judge_model"
    else:
        target = "policy_model"
    if capability == "single_turn":
        config.component_capabilities.policy_model = {"vision": True}
        dataset.required_harness_capabilities = [capability]
        target = "microvqa_simple_agent"
        config.component_capabilities[target] = {}
    config.policy_model.responses_api_models.openai_model.default_responses_create_params = {
        "temperature": 0.1,
        "max_output_tokens": 16384,
    }
    if declared is not None:
        config.component_capabilities[target][capability] = declared
    file = tmp_path / "cell.yaml"
    OmegaConf.save(config, file)
    out = tmp_path / "plan.json"
    cmd = [
        str(Path(sys.executable).with_name("gym")),
        "eval",
        "plan",
        "--config",
        str(file),
        "--out",
        str(out),
        "--validate-compatibility",
    ]
    run = subprocess.run(
        cmd, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True, timeout=60
    )
    evidence = os.environ.get("MATRIX_EVIDENCE")
    if evidence:
        dest = Path(evidence)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{benchmark}-{capability}-{declared}.json").write_text(
            json.dumps(
                {
                    "command": cmd,
                    "exit_code": run.returncode,
                    "config": OmegaConf.to_container(config),
                    "config_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                    "sample_sha256": before,
                    "scorer_sha256": hashlib.sha256(
                        (ROOT / f"resources_servers/{benchmark}/app.py").read_bytes()
                    ).hexdigest(),
                    "harness_sha256": hashlib.sha256(
                        (ROOT / "responses_api_agents/simple_agent/app.py").read_bytes()
                    ).hexdigest(),
                    "stdout": run.stdout,
                    "stderr": run.stderr,
                    "plan": json.loads(out.read_text()) if out.exists() else None,
                },
                indent=2,
            )
        )
    assert hashlib.sha256(sample.read_bytes()).hexdigest() == before
    if declared is True:
        assert run.returncode == 0, run.stdout + run.stderr
        assert len(json.loads(out.read_text())["tasks"]) > 0
    else:
        assert run.returncode != 0
        assert not out.exists()
        assert ("unsupported" if declared is False else "undeclared") in run.stdout + run.stderr


def test_invalid_contract_rejected_before_preparation(tmp_path, monkeypatch):
    from nemo_gym.cli import eval as cli

    config = OmegaConf.create(
        {
            "plan_output_fpath": str(tmp_path / "must-not-exist.json"),
            "validate_plan_compatibility": True,
            "agent": {
                "responses_api_agents": {
                    "simple_agent": {
                        "model_server": {"type": "responses_api_models", "name": "model"},
                        "datasets": [
                            {
                                "name": "contract",
                                "type": "benchmark",
                                "jsonl_fpath": "absent.jsonl",
                                "prepare_script": "absent.py",
                                "required_model_capabilities": ["vision"],
                            }
                        ],
                    }
                }
            },
            "model": {"responses_api_models": {"openai_model": {}}},
            "component_capabilities": {"model": {"vision": False}},
        }
    )
    monkeypatch.setattr(cli, "get_global_config_dict", lambda **kwargs: config)
    monkeypatch.setattr(
        cli, "_prepare_datasets", lambda *args: pytest.fail("Preparation ran before rejected contract")
    )
    with pytest.raises(SystemExit):
        cli.plan_benchmark()
    assert not (tmp_path / "must-not-exist.json").exists()
