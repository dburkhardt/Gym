# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration-only model switching; no inference or extra model authorization."""

from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.provider_dispatch import configuration_digest
from responses_api_models.openai_model.app import SimpleModelServerConfig


def test_model_switch_preserves_native_benchmark_and_scorer():
    root = Path(__file__).resolve().parents[2]
    benchmark_path = root / "benchmarks/proteinlmbench/config.yaml"
    resource_path = root / "resources_servers/proteinlmbench/configs/proteinlmbench.yaml"
    scorer_path = root / "resources_servers/proteinlmbench/app.py"
    before = {path: path.read_bytes() for path in [benchmark_path, resource_path, scorer_path]}
    configurations = []
    bindings = []
    for model, endpoint in [
        ("contract-model-a", "http://127.0.0.1:1/v1"),
        ("contract-model-b", "http://127.0.0.1:2/v1"),
    ]:
        raw = OmegaConf.merge(
            OmegaConf.load(resource_path),
            OmegaConf.load(benchmark_path),
            OmegaConf.load(root / "responses_api_models/openai_model/configs/openai_model.yaml"),
            {"policy_model_name": model, "policy_base_url": endpoint, "policy_api_key": "fixture-only"},
        )
        resolved = GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=raw,
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
            )
        )
        component = OmegaConf.to_container(resolved.policy_model.responses_api_models.openai_model, resolve=True)
        config = SimpleModelServerConfig.model_validate({**component, "name": "policy_model"})
        assert config.openai_model == model
        assert config.openai_base_url == endpoint
        binding = BenchmarkConfig.from_initial_config_dict(benchmark_path, raw)
        bindings.append((binding.name, binding.agent_name, binding.dataset.model_dump()))
        configurations.append(configuration_digest({"model": model, "endpoint": endpoint}))
    assert bindings[0] == bindings[1]
    assert configurations[0] != configurations[1]
    assert all(path.read_bytes() == contents for path, contents in before.items())
