# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from omegaconf import OmegaConf

from nemo_gym.config_types import ConfigError
from nemo_gym.eval_compatibility import validate_eval_compatibility


def configuration(model=None, harness=None):
    return OmegaConf.create(
        {
            "resource": {
                "resources_servers": {
                    "fixture": {
                        "datasets": [
                            {
                                "name": "vision-tools",
                                "type": "benchmark",
                                "jsonl_fpath": "data.jsonl",
                                "prepare_script": "prepare.py",
                                "required_model_capabilities": ["vision"],
                                "required_harness_capabilities": ["tools"],
                            }
                        ]
                    }
                }
            },
            "agent": {
                "responses_api_agents": {
                    "fixture": {
                        "resources_server": {"type": "resources_servers", "name": "resource"},
                        "model_server": {"type": "responses_api_models", "name": "solver"},
                    }
                }
            },
            "solver": {"responses_api_models": {"fixture": {}}},
            "component_capabilities": {"solver": model or {}, "agent": harness or {}},
        }
    )


def test_declared_compatible_pair():
    validate_eval_compatibility(configuration({"vision": True}, {"tools": True}))


@pytest.mark.parametrize("value,description", [(False, "unsupported"), (None, "undeclared"), ("true", "undeclared")])
def test_model_rejection_is_explicit(value, description):
    with pytest.raises(ConfigError, match=f"model 'solver' capability 'vision' is {description}"):
        validate_eval_compatibility(configuration({"vision": value}, {"tools": True}))


def test_harness_rejection():
    with pytest.raises(ConfigError, match="harness 'agent' capability 'tools' is undeclared"):
        validate_eval_compatibility(configuration({"vision": True}))


def test_declaration_on_different_model_does_not_authorize_solver():
    config = configuration(harness={"tools": True})
    config.component_capabilities.other_model = {"vision": True}
    with pytest.raises(ConfigError, match="model 'solver'.*undeclared"):
        validate_eval_compatibility(config)


@pytest.mark.parametrize("broken", ["missing", "wrong_type", "wrong_block"])
def test_invalid_model_binding_fails_even_with_capability_declaration(broken):
    config = configuration({"vision": True}, {"tools": True})
    if broken == "missing":
        del config.solver
    elif broken == "wrong_type":
        config.agent.responses_api_agents.fixture.model_server.type = "resources_servers"
    else:
        config.solver = {"resources_servers": {"fixture": {}}}
    with pytest.raises(ConfigError, match="valid model_server binding"):
        validate_eval_compatibility(config)


@pytest.mark.parametrize(
    "name,capability",
    [
        ("microvqa", "vision"),
        ("reactbench", "vision"),
        ("microvqa_plusplus_demo", "vision"),
        ("brainbench", "conditional_loglikelihood"),
    ],
)
def test_native_benchmark_requirements_fail_closed(name, capability):
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    benchmark = OmegaConf.load(root / "benchmarks" / name / "config.yaml")
    dependencies = [OmegaConf.load(root / p) for p in benchmark.get("config_paths", [])]
    config = OmegaConf.merge(*dependencies, benchmark, {"policy_model": {"responses_api_models": {"fixture": {}}}})
    with pytest.raises(ConfigError, match=f"capability '{capability}' is undeclared"):
        validate_eval_compatibility(config)
    config.component_capabilities = {"policy_model": {capability: False}}
    with pytest.raises(ConfigError, match="unsupported"):
        validate_eval_compatibility(config)
    config.component_capabilities.policy_model[capability] = True
    validate_eval_compatibility(config)


@pytest.mark.parametrize("entrypoint", ["e2e_rollout_collection", "collect_rollouts"])
def test_native_entrypoints_reject_before_execution(monkeypatch, capsys, entrypoint):
    from nemo_gym import rollout_collection
    from nemo_gym.cli import eval as cli

    monkeypatch.setattr(cli, "get_global_config_dict", lambda: configuration(harness={"tools": True}))

    def forbidden(*args, **kwargs):
        pytest.fail("Execution started before rejecting incompatible configuration")

    monkeypatch.setattr(cli.RunHelper, "start", forbidden)
    monkeypatch.setattr(rollout_collection.RolloutCollectionHelper, "__init__", forbidden)
    with pytest.raises(SystemExit) as exc:
        getattr(cli, entrypoint)()
    assert exc.value.code == 1
    assert "vision" in capsys.readouterr().out
