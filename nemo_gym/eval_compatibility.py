# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit native configuration contracts, not an empirical capability registry."""

from omegaconf import DictConfig

from nemo_gym.config_types import BenchmarkDatasetConfig, ConfigError
from nemo_gym.global_config import get_first_server_config_dict, resolve_dataset_agent


def validate_eval_compatibility(config: DictConfig) -> None:
    declarations = config.get("component_capabilities", {})
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
            if not (dataset.required_model_capabilities or dataset.required_harness_capabilities):
                continue
            agent = resolve_dataset_agent(config, owner, pin=dataset.agent, taskset=dataset.taskset)
            if agent is None:
                raise ConfigError(f"{dataset.name}: capability validation requires an explicit agent binding")
            agent_config = get_first_server_config_dict(config, agent)
            model_ref = agent_config.get("model_server") or {}
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
                for capability in required:
                    value = supported.get(capability)
                    if value is not True:
                        disposition = "unsupported" if value is False else "undeclared"
                        raise ConfigError(
                            f"{dataset.name}: {role} {instance!r} capability {capability!r} is {disposition}"
                        )
