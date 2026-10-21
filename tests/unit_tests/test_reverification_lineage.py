# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

import nemo_gym.rollout_reverification as rr


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "plugin"
    component = root / "resources_servers/scorer"
    component.mkdir(parents=True)
    (component / "requirements.txt").write_text("")
    (component / "app.py").write_text("VALUE = 1\n")
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(root))
    inputs, rollouts = tmp_path / "inputs.jsonl", tmp_path / "rollouts.jsonl"
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0, "reference": "A"}
    inputs.write_text(json.dumps(row) + "\n")
    rollouts.write_text(json.dumps(row | {"response": {"output": []}}) + "\n")
    global_config = OmegaConf.create(
        {
            "scorer": {"resources_servers": {"scorer": {"entrypoint": "app.py", "threshold": 1, "port": 1234}}},
            "policy_api_key": "SECRET",
            "head_server": {"port": 12},
        }
    )
    monkeypatch.setattr(rr, "setup_server_client", lambda: SimpleNamespace(global_config_dict=global_config))
    monkeypatch.setattr(rr, "_guard_reverify_mode", AsyncMock(return_value=None))
    monkeypatch.setattr(rr, "_validate_reverification_guard", lambda expected: None)
    config = rr.RolloutReverificationConfig(
        materialized_inputs_jsonl_fpath=str(inputs),
        rollouts_jsonl_fpath=str(rollouts),
        output_jsonl_fpath=str(tmp_path / "out.jsonl"),
    )
    return config, global_config, component, inputs


@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize("change", ["reference", "scorer", "config"])
async def test_changed_lineage_refuses_before_cache_or_dispatch(setup, monkeypatch, guarded, change):
    config, global_config, component, inputs = setup
    config.provider_dispatch_guard = guarded
    output = Path(config.output_jsonl_fpath)
    binding = rr._verification_binding(config)
    rr._verification_run_identity(output, resume=False, append=False, overwrite=False, binding=binding)
    output.write_text('{"_ng_task_index":0,"_ng_rollout_index":0,"reward":1}\n')
    original = {p: p.read_bytes() for p in output.parent.glob("out*")}
    if change == "reference":
        inputs.write_text(inputs.read_text().replace('"A"', '"B"'))
    elif change == "scorer":
        (component / "app.py").write_text("VALUE = 2\n")
    else:
        global_config.scorer.resources_servers.scorer.threshold = 2
    config.resume_from_cache = True
    inner = AsyncMock()
    monkeypatch.setattr(rr.RolloutReverificationHelper, "_run_from_config", inner)
    monkeypatch.setattr(
        rr, "_load_cache_keys_by_status", lambda *a, **kw: pytest.fail("cache examined before mismatch refusal")
    )
    with pytest.raises(rr.ConfigError, match="lineage changed"):
        await rr.RolloutReverificationHelper().run_from_config(config)
    inner.assert_not_called()
    assert all(p.read_bytes() == data for p, data in original.items())


@pytest.mark.parametrize("guarded", [False, True])
async def test_unchanged_resume_preserves_identity_and_placement_not_science(setup, monkeypatch, guarded):
    config, global_config, _, _ = setup
    config.provider_dispatch_guard = guarded
    output = Path(config.output_jsonl_fpath)
    initial = rr._verification_binding(config)
    identity = rr._verification_run_identity(output, resume=False, append=False, overwrite=False, binding=initial)
    global_config.scorer.resources_servers.scorer.port = 9999
    global_config.head_server.port = 9998
    global_config.policy_api_key = "ROTATED_SECRET"
    config.num_samples_in_parallel = 2
    config.resume_from_cache = True
    assert rr._verification_binding(config) == initial
    inner = AsyncMock(return_value=[])
    monkeypatch.setattr(rr.RolloutReverificationHelper, "_run_from_config", inner)
    await rr.RolloutReverificationHelper().run_from_config(config)
    assert inner.call_args.kwargs["verification_run_id"] == identity
    raw = output.with_name(output.name + ".reverify-identity.json").read_text()
    assert "SECRET" not in raw and str(output.parent) not in raw
    assert "external_assets_packages_and_services_unattested" in raw


def test_legacy_and_fresh_output(setup):
    config, _, _, _ = setup
    output = Path(config.output_jsonl_fpath)
    rr._verification_run_identity(output, resume=False, append=False, overwrite=False)
    old = output.with_name(output.name + ".reverify-identity.json").read_bytes()
    with pytest.raises(rr.ConfigError, match="legacy binding missing"):
        rr._verification_run_identity(
            output, resume=True, append=False, overwrite=False, binding=rr._verification_binding(config)
        )
    assert output.with_name(output.name + ".reverify-identity.json").read_bytes() == old
    fresh = output.with_name("fresh.jsonl")
    assert rr._verification_run_identity(
        fresh, resume=False, append=False, overwrite=False, binding=rr._verification_binding(config)
    )


@pytest.mark.parametrize("guarded", [False, True])
async def test_public_fresh_then_resume_keeps_cached_score_and_reference(setup, monkeypatch, guarded):
    config, _, _, inputs = setup
    config.provider_dispatch_guard = guarded
    config.disable_aggregation = True
    monkeypatch.setattr(rr, "get_exporters", list)
    dispatched = []

    async def verify(row):
        dispatched.append(row["verification_id"])
        return row, dict(row, reward=1.0)

    monkeypatch.setattr(rr, "_run_verification_payloads", lambda rows, **kw: [verify(row) for row in rows])
    helper = rr.RolloutReverificationHelper()
    first = await helper.run_from_config(config)
    assert len(first) == len(dispatched) == 1
    output = Path(config.output_jsonl_fpath)
    raw = output.read_bytes()
    config.resume_from_cache = True
    resumed = await helper.run_from_config(config)
    assert resumed == first and output.read_bytes() == raw and len(dispatched) == 1
    inputs.write_text(inputs.read_text().replace('"A"', '"B"'))
    with pytest.raises(rr.ConfigError, match="lineage changed"):
        await helper.run_from_config(config)
    assert output.read_bytes() == raw and len(dispatched) == 1


def test_nested_tool_host_port_are_scientific(setup):
    config, global_config, _, _ = setup
    settings = global_config.scorer.resources_servers.scorer
    settings.tool = {"host": "scientific-service-a", "port": 123}
    before = rr._verification_binding(config)
    settings.tool.host = "scientific-service-b"
    assert rr._verification_binding(config) != before
    settings.tool.host = "scientific-service-a"
    settings.tool.port = 456
    assert rr._verification_binding(config) != before


def test_identical_component_relocation_preserves_binding(setup, monkeypatch, tmp_path):
    import shutil

    config, _, component, _ = setup
    before = rr._verification_binding(config)
    relocated = tmp_path / "z-other-placement"
    shutil.copytree(component, relocated)
    monkeypatch.setattr("nemo_gym.cli.env._resolve_server_dir", lambda path: relocated)
    assert rr._verification_binding(config) == before
