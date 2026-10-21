# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json

import pytest

from nemo_gym import mixed_scientific_restart as mixed
from nemo_gym.config_types import ConfigError


def fixture():
    def ref(name, kind):
        return {"name": name, "type": kind}

    config = {"model": {"responses_api_models": {"openai_model": {}}}}
    for prefix, agent, resource in [
        ("sci", "scicode_agent", "scicode"),
        ("mol", "simple_agent", "campaign_creator_scoring"),
    ]:
        config[prefix + "_agent"] = {
            "responses_api_agents": {
                agent: {
                    "model_server": ref("model", "responses_api_models"),
                    "resources_server": ref(prefix + "_resource", "resources_servers"),
                    "max_steps": 1,
                }
            }
        }
        config[prefix + "_resource"] = {"resources_servers": {resource: {"benchmark": "moljson"}}}
        config[prefix + "_relay"] = {
            "environment_servers": {"legacy_agent": {"agent_server": ref(prefix + "_agent", "responses_api_agents")}}
        }
    rows = [
        {"agent_ref": {"name": "sci_agent"}, "task_source": "sci_agent"},
        {"agent_ref": {"name": "mol_agent"}, "task_source": "mol_resource"},
    ]
    return config, rows


def test_exact_native_routes_and_shared_model():
    config, rows = fixture()
    sci, creator, subset, routes = mixed.partition_routes(config, rows)
    assert set(sci) == {"model", "sci_agent", "sci_resource", "sci_relay"}
    assert set(creator) == {"model", "mol_agent", "mol_resource", "mol_relay"}
    assert subset == [rows[1]]
    assert routes == [
        {"agent": "sci_agent", "source": "sci_agent", "relay": "sci_relay"},
        {"agent": "mol_agent", "source": "mol_resource", "relay": "mol_relay"},
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "agent",
        "relay",
        "missing",
        "unused",
        "extra",
        "shared_resource",
        "two_relays",
        "different_model",
        "wrong_type",
        "malformed",
    ],
)
def test_refuses_unqualified_routes(mutation):
    config, rows = fixture()
    if mutation == "source":
        rows[0]["task_source"] = "mol_resource"
    elif mutation == "agent":
        rows[0]["agent_ref"]["name"] = "mol_agent"
    elif mutation == "relay":
        rows[0]["_ng_environment_server"] = "mol_relay"
    elif mutation == "missing":
        rows[0].pop("agent_ref")
    elif mutation == "unused":
        rows.pop()
    elif mutation == "extra":
        config["extra"] = {"resources_servers": {"other": {}}}
    elif mutation == "shared_resource":
        config["mol_agent"]["responses_api_agents"]["simple_agent"]["resources_server"]["name"] = "sci_resource"
    elif mutation == "two_relays":
        config["duplicate_relay"] = copy.deepcopy(config["sci_relay"])
    elif mutation == "different_model":
        config["second_model"] = copy.deepcopy(config["model"])
    elif mutation == "wrong_type":
        rows[0]["agent_ref"]["type"] = "resources_servers"
    else:
        rows[0] = "malformed"
    with pytest.raises(ConfigError):
        mixed.partition_routes(config, rows)


@pytest.mark.parametrize("fail", [False, True])
def test_delegates_exact_closures_and_cleans_temporary_subset(tmp_path, monkeypatch, fail):
    config, rows = fixture()
    path = tmp_path / "all.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    seen = []
    monkeypatch.setattr(
        mixed, "qualify_scicode_dependencies", lambda root, cfg: {"actual_scicode_closure": sorted(cfg)}
    )

    def creator(root, cfg, subset):
        seen.append(subset)
        assert [json.loads(x) for x in subset.read_text().splitlines()] == [rows[1]]
        if fail:
            raise ConfigError("dependency unavailable")
        return {"materialized_inputs_sha256": hashlib.sha256(subset.read_bytes()).hexdigest()}

    monkeypatch.setattr(mixed, "qualify_creator_dependencies", creator)
    if fail:
        with pytest.raises(ConfigError):
            mixed.qualify_mixed_scientific_dependencies(tmp_path, config, path)
    else:
        result = mixed.qualify_mixed_scientific_dependencies(tmp_path, config, path)
        assert result["materialized_inputs_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert str(seen[0]) not in json.dumps(result)
        assert result["tasks"] == 2
    assert len(seen) == 1 and not seen[0].exists() and not seen[0].parent.exists()
    assert len(path.read_text().splitlines()) == 2


@pytest.mark.parametrize("unsupported", [False, True])
def test_actual_binder_selects_mixed_without_pure_overwrite(tmp_path, monkeypatch, unsupported):
    from types import SimpleNamespace

    import nemo_gym.creator_restart as creator_module
    import nemo_gym.scicode_restart as sci_module
    from nemo_gym.batch_restart import bind_execution

    config, rows = fixture()
    for name in ("sci_relay", "mol_relay"):
        config[name]["environment_servers"]["legacy_agent"]["agent_server"]["type"] = "responses_api_agents"
    path = tmp_path / "materialized.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    marker = {"protocol": "actual-mixed-selection-marker", "both_closures": ["sci", "mol"]}
    calls = []

    def qualify(root, actual_config, actual_path):
        assert actual_config == config and actual_path == path
        calls.append(True)
        if unsupported:
            raise ConfigError("mixed dependency unavailable")
        return marker

    def pure(*args):
        raise AssertionError("Pure closure must never overwrite mixed qualification")

    monkeypatch.setattr(mixed, "qualify_mixed_scientific_dependencies", qualify)
    monkeypatch.setattr(creator_module, "qualify_creator_dependencies", pure)
    monkeypatch.setattr(sci_module, "qualify_scicode_dependencies", pure)
    tracker = SimpleNamespace(manifest_sha256="m" * 64, execution_binding=None)
    bind_execution(tracker, config, path, resume=False, require_prior=False, capture_dirs=[tmp_path / "capture"])
    assert tracker.execution_binding["restart_scope_qualified"] is (not unsupported)
    # The complete mixed closure is retained, rather than a later pure branch.
    closure = tracker.execution_binding["scientific_dependencies"]
    if unsupported:
        assert closure == {"unsupported": "mixed dependency unavailable"}
        with pytest.raises(ConfigError, match="mixed dependency unavailable"):
            bind_execution(tracker, config, path, resume=True, require_prior=True, capture_dirs=[tmp_path / "capture"])
        assert calls == [True, True]
    else:
        assert closure == marker
        assert calls == [True]
