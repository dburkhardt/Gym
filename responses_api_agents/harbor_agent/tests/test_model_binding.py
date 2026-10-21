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
