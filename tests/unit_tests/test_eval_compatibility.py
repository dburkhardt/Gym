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
        ("brainbench", "whole_chat_prompt_loglikelihood"),
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


def role_requirement(component, field, capability="text_generation"):
    return {"component": component, "binding_field": field, "capabilities": [capability]}


def role_config():
    config = configuration({"vision": True}, {"tools": True})
    config.resource.resources_servers.fixture.judge_model_server = {"type": "responses_api_models", "name": "judge"}
    config.judge = {"responses_api_models": {"fixture": {}}}
    config.resource.resources_servers.fixture.datasets[0].required_model_roles = {
        "judge": role_requirement("resource", "judge_model_server")
    }
    return config


@pytest.mark.parametrize("capability", ["vision", "tools", "conditional_loglikelihood"])
def test_role_capabilities_do_not_borrow_solver_declarations(capability):
    config = role_config()
    config.resource.resources_servers.fixture.datasets[0].required_model_roles.judge.capabilities = [capability]
    config.component_capabilities.solver[capability] = True
    with pytest.raises(ConfigError, match="role 'judge'.*undeclared"):
        validate_eval_compatibility(config)
    config.component_capabilities.judge = {capability: False}
    with pytest.raises(ConfigError, match="role 'judge'.*unsupported"):
        validate_eval_compatibility(config)
    config.component_capabilities.judge[capability] = True
    validate_eval_compatibility(config)


@pytest.mark.parametrize("broken", ["missing_field", "wrong_ref", "missing_target", "wrong_target", "unreachable"])
def test_role_binding_fails_closed(broken):
    config = role_config()
    config.component_capabilities.judge = {"text_generation": True}
    component = config.resource.resources_servers.fixture
    if broken == "missing_field":
        del component.judge_model_server
    elif broken == "wrong_ref":
        component.judge_model_server.type = "resources_servers"
    elif broken == "missing_target":
        del config.judge
    elif broken == "wrong_target":
        config.judge = {"resources_servers": {"fixture": {}}}
    else:
        config.unused = {"resources_servers": {"fixture": {"judge_model_server": component.judge_model_server}}}
        component.datasets[0].required_model_roles.judge.component = "unused"
    with pytest.raises(ConfigError):
        validate_eval_compatibility(config)


def test_role_tracks_actual_ref_override_and_shared_model():
    config = role_config()
    config.component_capabilities.judge = {"text_generation": True}
    config.resource.resources_servers.fixture.judge_model_server.name = "solver"
    with pytest.raises(ConfigError, match="model 'solver'.*undeclared"):
        validate_eval_compatibility(config)
    config.component_capabilities.solver.text_generation = True
    config.resource.resources_servers.fixture.datasets[0].required_model_roles.helper = role_requirement(
        "agent", "model_server", "vision"
    )
    validate_eval_compatibility(config)


@pytest.mark.parametrize("kind", ["biomaze", "policy_helper"])
def test_actual_native_role_fields(kind):
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if kind == "biomaze":
        config = OmegaConf.load(root / "resources_servers/biomaze/configs/biomaze.yaml")
        owner = "biomaze"
        component = config.biomaze.resources_servers.biomaze
        component.judge_model_server = {"type": "responses_api_models", "name": "judge"}
        roles = {"conclusion_judge": role_requirement(owner, "judge_model_server")}
    else:
        config = OmegaConf.load(
            root
            / "responses_api_agents/conversational_tool_use_policy_tool_generation/configs/conversational_tool_use_policy_tool_generation.yaml"
        )
        owner = "conversational_tool_use_policy_tool_generation"
        component = config[owner].responses_api_agents[owner]
        roles = {
            "policy_helper": role_requirement(owner, "policy_model_server"),
            "cohesion_judge": role_requirement(owner, "judge_model_server"),
        }
        config.policy_generation_model = {"responses_api_models": {"fixture": {}}}
        config.policy_tool_judge_model = {"responses_api_models": {"fixture": {}}}
    component.datasets = [
        {
            "name": "actual-role-contract",
            "type": "benchmark",
            "jsonl_fpath": "unused.jsonl",
            "prepare_script": "unused.py",
            "required_model_roles": roles,
        }
    ]
    config.policy_model = {"responses_api_models": {"fixture": {}}}
    config.judge = {"responses_api_models": {"fixture": {}}}
    config.component_capabilities = {}
    with pytest.raises(ConfigError, match="undeclared"):
        validate_eval_compatibility(config)
    for name in ["judge", "policy_generation_model", "policy_tool_judge_model"]:
        config.component_capabilities[name] = {"text_generation": True}
    validate_eval_compatibility(config)


@pytest.mark.parametrize("entrypoint", ["e2e_rollout_collection", "collect_rollouts"])
def test_native_entrypoints_reject_unsupported_judge_before_execution(monkeypatch, capsys, entrypoint):
    from nemo_gym import rollout_collection
    from nemo_gym.cli import eval as cli

    config = role_config()
    config.component_capabilities.judge = {"text_generation": False}
    monkeypatch.setattr(cli, "get_global_config_dict", lambda: config)

    def forbidden(*args, **kwargs):
        pytest.fail("Execution started before rejecting incompatible judge role")

    monkeypatch.setattr(cli.RunHelper, "start", forbidden)
    monkeypatch.setattr(rollout_collection.RolloutCollectionHelper, "__init__", forbidden)
    with pytest.raises(SystemExit) as error:
        getattr(cli, entrypoint)()
    assert error.value.code == 1
    output = capsys.readouterr().out
    assert "role 'judge'" in output and "unsupported" in output


@pytest.mark.parametrize("value", [None, "true", 1])
def test_role_capability_requires_literal_boolean(value):
    config = role_config()
    config.component_capabilities.judge = {"text_generation": value}
    with pytest.raises(ConfigError, match="undeclared"):
        validate_eval_compatibility(config)


@pytest.mark.parametrize("field", ["model_server", "resources_server"])
def test_role_only_malformed_agent_reference_is_config_error(field):
    config = role_config()
    dataset = config.resource.resources_servers.fixture.datasets[0]
    dataset.required_model_capabilities = []
    dataset.required_harness_capabilities = []
    # Put the dataset on the selected agent so malformed resource binding does
    # not fail an unrelated owner-to-agent routing lookup first.
    config.agent.responses_api_agents.fixture.datasets = [dataset]
    config.resource.resources_servers.fixture.datasets = []
    config.agent.responses_api_agents.fixture[field] = "not-a-reference"
    with pytest.raises(ConfigError, match="reference mapping"):
        validate_eval_compatibility(config)
