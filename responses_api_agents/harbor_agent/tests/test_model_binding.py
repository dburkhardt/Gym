# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from responses_api_agents.harbor_agent.app import HarborAgent


def resolve(service, alias=None, backend="openai_model"):
    config = {"selected": {"responses_api_models": {backend: service}}}
    if alias is not None:
        config["policy_model_name"] = alias
    agent = SimpleNamespace(config=SimpleNamespace(model_server=SimpleNamespace(name="selected")))
    return HarborAgent._resolve_model_name(agent, config)


def test_selected_service_identity_without_global_alias():
    assert resolve({"openai_model": "authorized-model"}) == "authorized-model"
    assert resolve({"model": "provider-model"}, backend="inference_provider") == "provider-model"
    assert resolve({"openai_model": "switched-model"}) == "switched-model"


def test_compatible_legacy_alias():
    assert resolve({}, "legacy-model") == "legacy-model"
    assert resolve({"openai_model": "same"}, "same") == "same"


@pytest.mark.parametrize(
    "service,alias",
    [
        ({}, None),
        ({"openai_model": "a"}, "b"),
        ({"openai_model": "a", "model": "b"}, None),
        ({"openai_model": ""}, None),
        ({"model": 3}, None),
    ],
)
def test_conflicting_or_missing_identity_refused(service, alias):
    with pytest.raises(ValueError):
        resolve(service, alias)


from pathlib import Path

from nemo_gym.server_utils import rollout_path_prefix
from responses_api_agents.harbor_agent.tests.test_app import _make_server


def test_independent_judge_route_preserves_solver_and_capture():
    model = "nvidia/nemotron-3.5-super"
    server = _make_server(
        judge_model_server={"type": "responses_api_models", "name": "judge"},
        harbor_verifier_import_path="benchmarks.naturebench.verifier:NatureBenchVerifier",
    )
    config = {
        "judge": {
            "responses_api_models": {"openai_model": {"openai_model": model, "host": "judge-host", "port": 9002}}
        },
        "provider_dispatch_guard": True,
    }
    route = server._resolve_judge_route(config, "rollout-123")
    expected = "http://judge-host:9002" + rollout_path_prefix("rollout-123") + "/v1"
    assert route == (expected, model)
    job = server._build_job_config(
        "scientific",
        "task",
        model,
        "http://solver:9001/v1",
        "test",
        Path("/tmp/jobs"),
        judge_route=route,
        gym_provider_dispatch_guard=True,
    )
    assert job["agents"][0]["kwargs"]["api_base"] == "http://solver:9001/v1"
    assert job["verifier"]["kwargs"]["gym_model_api_base"] == expected
    assert job["verifier"]["kwargs"]["gym_model_name"] == model
    assert job["verifier"]["kwargs"]["gym_provider_dispatch_guard"] is True
    legacy = server._build_job_config("scientific", "task", model, "http://solver:9001/v1", "test", Path("/tmp/jobs"))
    assert legacy["verifier"]["kwargs"]["gym_model_api_base"] == "http://solver:9001/v1"
    assert _make_server()._resolve_judge_route({}, None) is None


@pytest.mark.parametrize(
    "service",
    [
        None,
        {},
        {"openai_model": ""},
        {"openai_model": "a", "model": "b", "host": "localhost", "port": 1234},
        {"openai_model": "a", "host": "localhost", "port": -1},
    ],
)
def test_invalid_judge_binding_fails_closed(service):
    server = _make_server(
        judge_model_server={"type": "responses_api_models", "name": "judge"},
        harbor_verifier_import_path="benchmarks.naturebench.verifier:NatureBenchVerifier",
    )
    config = {} if service is None else {"judge": {"responses_api_models": {"openai_model": service}}}
    with pytest.raises(ValueError):
        server._resolve_judge_route(config, "rollout")


def test_guarded_judge_requires_capture_and_custom_verifier():
    server = _make_server(judge_model_server={"type": "responses_api_models", "name": "judge"})
    with pytest.raises(ValueError, match="custom Harbor verifier"):
        server._resolve_judge_route({}, None)
    server.config.harbor_verifier_import_path = "custom:Verifier"
    config = {
        "judge": {
            "responses_api_models": {"openai_model": {"openai_model": "model", "host": "localhost", "port": 1234}}
        },
        "provider_dispatch_guard": True,
    }
    with pytest.raises(ValueError, match="capture identity"):
        server._resolve_judge_route(config, None)


@pytest.mark.parametrize(
    "target", [None, [], 3, {"resources_servers": {"fixture": {"model": "model", "host": "localhost", "port": 1234}}}]
)
def test_judge_reference_rejects_non_model_service(target):
    server = _make_server(
        judge_model_server={"type": "responses_api_models", "name": "judge"},
        harbor_verifier_import_path="custom:Verifier",
    )
    with pytest.raises(ValueError, match="model service"):
        server._resolve_judge_route({"judge": target}, "rollout")
