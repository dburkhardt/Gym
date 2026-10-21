# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit native configuration contracts, not an empirical capability registry."""

from omegaconf import DictConfig
from pydantic import ValidationError

from nemo_gym.config_types import BenchmarkDatasetConfig, ConfigError, ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict, resolve_dataset_agent


def validate_eval_compatibility(config: DictConfig) -> None:
    declarations = config.get("component_capabilities", {})
    if not isinstance(declarations, (dict, DictConfig)):
        raise ConfigError("component_capabilities must be a mapping")
    for owner, block in config.items():
        if not isinstance(block, (dict, DictConfig)):
            continue
        if not ("resources_servers" in block or "responses_api_agents" in block):
            continue
        inner = get_first_server_config_dict(config, owner)
        for raw in inner.get("datasets") or []:
            if raw.get("type") != "benchmark":
                continue
            dataset = BenchmarkDatasetConfig.model_validate(raw)
            if not (
                dataset.required_model_capabilities
                or dataset.required_harness_capabilities
                or dataset.required_model_roles
            ):
                continue
            agent = resolve_dataset_agent(config, owner, pin=dataset.agent, taskset=dataset.taskset)
            if agent is None:
                raise ConfigError(f"{dataset.name}: capability validation requires an explicit agent binding")
            agent_config = get_first_server_config_dict(config, agent)
            model_ref = agent_config.get("model_server") or {}
            if not isinstance(model_ref, (dict, DictConfig)):
                raise ConfigError(f"{dataset.name}: agent model_server must be a model reference mapping")
            model = model_ref.get("name")
            if dataset.required_model_capabilities and (
                model_ref.get("type") != "responses_api_models"
                or model not in config
                or not isinstance(config[model], (dict, DictConfig))
                or "responses_api_models" not in config[model]
            ):
                raise ConfigError(f"{dataset.name}: required model capabilities need a valid model_server binding")
            for role, instance, required in [
                ("harness", agent, dataset.required_harness_capabilities),
                ("model", model, dataset.required_model_capabilities),
            ]:
                supported = declarations.get(instance, {}) if instance else {}
                if not isinstance(supported, (dict, DictConfig)):
                    raise ConfigError(f"{dataset.name}: capabilities for {instance!r} must be a mapping")
                for capability in required:
                    value = supported.get(capability)
                    if value is not True:
                        disposition = "unsupported" if value is False else "undeclared"
                        raise ConfigError(
                            f"{dataset.name}: {role} {instance!r} capability {capability!r} is {disposition}"
                        )

            # Only the selected dataset owner/agent and its explicitly bound
            # resources server are admitted here. No name-based graph discovery.
            reachable = {owner, agent}
            resource_ref = agent_config.get("resources_server") or {}
            if not isinstance(resource_ref, (dict, DictConfig)):
                raise ConfigError(f"{dataset.name}: agent resources_server must be a resource reference mapping")
            if resource_ref.get("type") == "resources_servers":
                resource_name = resource_ref.get("name")
                if (
                    resource_name in config
                    and isinstance(config[resource_name], (dict, DictConfig))
                    and "resources_servers" in config[resource_name]
                ):
                    reachable.add(resource_name)
            for role, requirement in dataset.required_model_roles.items():
                if requirement.component not in reachable:
                    raise ConfigError(
                        f"{dataset.name}: role {role!r} component {requirement.component!r} is outside the selected runtime graph"
                    )
                component = get_first_server_config_dict(config, requirement.component)
                try:
                    reference = ModelServerRef.model_validate(component.get(requirement.binding_field))
                except ValidationError as error:
                    raise ConfigError(
                        f"{dataset.name}: role {role!r} requires a valid model reference at {requirement.component}.{requirement.binding_field}"
                    ) from error
                target = config.get(reference.name)
                if not isinstance(target, (dict, DictConfig)) or "responses_api_models" not in target:
                    raise ConfigError(
                        f"{dataset.name}: role {role!r} target {reference.name!r} is not a configured model server"
                    )
                supported = declarations.get(reference.name, {})
                if not isinstance(supported, (dict, DictConfig)):
                    raise ConfigError(f"{dataset.name}: role {role!r} capabilities must be a mapping")
                for capability in requirement.capabilities:
                    value = supported.get(capability)
                    if value is not True:
                        disposition = "unsupported" if value is False else "undeclared"
                        raise ConfigError(
                            f"{dataset.name}: role {role!r} model {reference.name!r} capability {capability!r} is {disposition}"
                        )
