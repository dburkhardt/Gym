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


def test_solver_switch_preserves_biomaze_judge_sample_and_scoring():
    """A solver comparison does not silently change the independently bound judge."""
    from resources_servers.biomaze.app import Config as BioMazeConfig

    root = Path(__file__).resolve().parents[2]
    benchmark = root / "benchmarks/biomaze/config.yaml"
    resource = root / "resources_servers/biomaze/configs/biomaze.yaml"
    files = [
        benchmark,
        resource,
        root / "resources_servers/biomaze/app.py",
        root / "resources_servers/biomaze/openended.py",
    ]
    before = {path: path.read_bytes() for path in files}
    comparisons = []
    for solver, endpoint in [
        ("contract-solver-a", "http://127.0.0.1:1/v1"),
        ("contract-solver-b", "http://127.0.0.1:2/v1"),
    ]:
        raw = OmegaConf.merge(
            OmegaConf.load(resource),
            OmegaConf.load(benchmark),
            OmegaConf.load(root / "responses_api_models/openai_model/configs/openai_model.yaml"),
            {"policy_model_name": solver, "policy_base_url": endpoint, "policy_api_key": "fixture-only"},
        )
        raw.judge_model = {
            "responses_api_models": {
                "openai_model": {
                    "entrypoint": "app.py",
                    "openai_model": "nvidia/nvidia/nemotron-3.5-super-vl",
                    "openai_base_url": "http://127.0.0.1:3/v1",
                    "openai_api_key": "fixture-only",
                    "extra_body": {"temperature": 0.0},
                }
            }
        }
        raw.biomaze.resources_servers.biomaze.judge_model_server = {
            "type": "responses_api_models",
            "name": "judge_model",
        }
        raw.biomaze.resources_servers.biomaze.judge_prompt_file = "/fixture/fixed-creator-prompt.txt"
        resolved = GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=raw,
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
            )
        )
        component = OmegaConf.to_container(resolved.policy_model.responses_api_models.openai_model, resolve=True)
        policy = SimpleModelServerConfig.model_validate({**component, "name": "policy_model"})
        assert policy.openai_model == solver and policy.openai_base_url == endpoint
        judged = OmegaConf.to_container(resolved.judge_model.responses_api_models.openai_model, resolve=True)
        judge = SimpleModelServerConfig.model_validate({**judged, "name": "judge_model"})
        resource_config = OmegaConf.to_container(resolved.biomaze.resources_servers.biomaze, resolve=True)
        scorer = BioMazeConfig.model_validate({**resource_config, "name": "biomaze"})
        assert scorer.judge_model_server.name == "judge_model"
        assert scorer.judge_model == judge.openai_model
        assert resolved.biomaze_simple_agent.responses_api_agents.simple_agent.model_server.name == "policy_model"
        binding = BenchmarkConfig.from_initial_config_dict(benchmark, raw)
        comparisons.append((judge.model_dump(), resource_config, binding.dataset.model_dump()))
    assert comparisons[0] == comparisons[1]
    assert all(path.read_bytes() == content for path, content in before.items())
