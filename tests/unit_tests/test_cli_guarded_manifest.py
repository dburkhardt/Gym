# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from nemo_gym.cli.eval import _ensure_guarded_batch_manifest
from nemo_gym.config_types import ConfigError


def test_manifest_preserves_existing_and_binds_science(tmp_path):
    config = OmegaConf.create(
        {
            "provider_dispatch_guard": True,
            "model": {"responses_api_models": {"openai_model": {"openai_model": "fixture", "temperature": 0.1}}},
        }
    )
    rows = [
        {
            "agent_ref": {"name": "agent"},
            "task_source": "agent",
            "task_id": "original",
            "_ng_task_index": 0,
            "_ng_rollout_index": 0,
            "responses_create_params": {"input": []},
        }
    ]
    collector = SimpleNamespace(batch_manifest_fpath=None)
    output = tmp_path / "results.jsonl"
    _ensure_guarded_batch_manifest(config, collector, rows, output)
    from pathlib import Path

    path = Path(collector.batch_manifest_fpath)
    before = path.read_bytes()
    collector.batch_manifest_fpath = None
    config["resume_from_cache"] = True
    _ensure_guarded_batch_manifest(config, collector, rows, output)
    assert path.read_bytes() == before
    for key, value in [("openai_model", "other"), ("temperature", 0.5)]:
        altered = OmegaConf.create(OmegaConf.to_container(config))
        altered.model.responses_api_models.openai_model[key] = value
        collector.batch_manifest_fpath = None
        with pytest.raises(ConfigError, match="manifest differs"):
            _ensure_guarded_batch_manifest(altered, collector, rows, output)
        assert path.read_bytes() == before
    rows[0]["task_id"] = "changed"
    collector.batch_manifest_fpath = None
    with pytest.raises(ConfigError, match="manifest differs"):
        _ensure_guarded_batch_manifest(config, collector, rows, output)
    assert path.read_bytes() == before


def test_manifest_explicit_and_unguarded_unchanged(tmp_path):
    for guarded, explicit in [(False, None), (True, "operator-manifest.json")]:
        collector = SimpleNamespace(batch_manifest_fpath=explicit)
        _ensure_guarded_batch_manifest(
            OmegaConf.create({"provider_dispatch_guard": guarded}), collector, [], tmp_path / "result.jsonl"
        )
        assert collector.batch_manifest_fpath == explicit
    assert not list(tmp_path.iterdir())


def test_native_taskset_execution_remains_unqualified(tmp_path, capsys):
    collector = SimpleNamespace(batch_manifest_fpath=None)
    _ensure_guarded_batch_manifest(
        OmegaConf.create({"provider_dispatch_guard": True}),
        collector,
        [{"task_id": {"taskset": "native"}, "task_input": {}}],
        tmp_path / "result.jsonl",
    )
    assert collector.batch_manifest_fpath is None
    assert "unqualified" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_generated_manifest_uses_output_owned_directory(tmp_path):
    config = OmegaConf.create({"provider_dispatch_guard": True})
    rows = [{"agent_ref": {"name": "agent"}, "task_source": "agent", "_ng_task_index": 0, "_ng_rollout_index": 0}]
    for name in ("first", "second"):
        collector = SimpleNamespace(batch_manifest_fpath=None)
        _ensure_guarded_batch_manifest(config, collector, rows, tmp_path / f"{name}.jsonl")
        assert collector.batch_manifest_fpath == str(tmp_path / name / "batch_manifest.json")
