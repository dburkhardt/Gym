# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
from omegaconf import OmegaConf

from nemo_gym.cli import eval as cli
from nemo_gym.rollout_reverification import RolloutReverificationConfig


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_SERVICE_OWNERSHIP_MODE", "process")
    for name in ("inputs.jsonl", "rollouts.jsonl"):
        (tmp_path / name).write_text("{}\n")
    return OmegaConf.create(
        {
            "materialized_inputs_jsonl_fpath": str(tmp_path / "inputs.jsonl"),
            "rollouts_jsonl_fpath": str(tmp_path / "rollouts.jsonl"),
            "output_jsonl_fpath": str(tmp_path / "out" / "scores.jsonl"),
            "resource": {"resources_servers": {"test": {"entrypoint": "app.py", "api_key": "secret-value"}}},
        }
    )


def owner(config):
    return cli._reverification_service_ownership(config, RolloutReverificationConfig.model_validate(config))


def test_binding_hashes_inputs_and_hides_component_secret(config):
    first = owner(config)
    try:
        assert first.directory.name == "native_services"
        binding = first.binding
        assert "secret-value" not in json.dumps(binding)
        assert len(binding["inputs"]) == 2
        assert binding["operation"] == "reverify"
    finally:
        first.close()
    from pathlib import Path

    Path(config.rollouts_jsonl_fpath).write_text('{"changed":true}\n')
    second = owner(config)
    try:
        assert binding["inputs"] != second.binding["inputs"]
    finally:
        second.close()


@pytest.mark.parametrize("kind", ["directory", "symlink", "unsafe_symlink", "missing_input"])
def test_bad_paths_refused_before_ownership_created(config, kind):
    from pathlib import Path

    output = Path(config.output_jsonl_fpath)
    output.parent.mkdir()
    if kind == "directory":
        output.mkdir()
    elif kind == "symlink":
        output.symlink_to(config.rollouts_jsonl_fpath)
    elif kind == "unsafe_symlink":
        config.force = True
        output.with_name("unsafe_" + output.name).symlink_to(config.rollouts_jsonl_fpath)
    else:
        Path(config.materialized_inputs_jsonl_fpath).unlink()
    with pytest.raises(Exception):
        owner(config)
    assert not (output.with_suffix("") / "native_services").exists()


def test_force_binding_declares_both_possible_outputs(config):
    config.force = True
    lease = owner(config)
    try:
        assert lease.binding["possible_outputs"] == [
            config.output_jsonl_fpath,
            config.output_jsonl_fpath.replace("scores.jsonl", "unsafe_scores.jsonl"),
        ]
    finally:
        lease.close()


@pytest.mark.parametrize("guard", [False, True])
@pytest.mark.parametrize("failure", [None, "score", "startup"])
def test_native_entrypoint_attaches_owner_and_settles(config, monkeypatch, guard, failure):
    config.provider_dispatch_guard = guard
    monkeypatch.setattr(cli, "get_global_config_dict", lambda: config)
    events = []
    holder = {}

    class Helper:
        def start(self, arg):
            holder["lease"] = self._service_ownership
            assert not self._service_ownership.lock.closed
            events.append("start")
            if failure == "startup":
                # Existing RunHelper.start owns partial-start settlement.
                self.shutdown()
                raise RuntimeError("startup")

        def shutdown(self):
            events.append("shutdown")
            self._service_ownership.close()

    monkeypatch.setattr(cli, "RunHelper", Helper)

    async def score(self, cfg):
        events.append("score")
        if failure == "score":
            raise RuntimeError("score")
        return []

    from nemo_gym.rollout_reverification import RolloutReverificationHelper

    monkeypatch.setattr(RolloutReverificationHelper, "run_from_config", score)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            cli.reverify_rollouts()
    else:
        cli.reverify_rollouts()
    assert events == (["start", "shutdown"] if failure == "startup" else ["start", "score", "shutdown"])
    assert holder["lease"].lock.closed


def test_invalid_config_never_constructs_runtime(config, monkeypatch):
    del config.materialized_inputs_jsonl_fpath
    monkeypatch.setattr(cli, "get_global_config_dict", lambda: config)

    def forbidden():
        pytest.fail("runtime constructed before config validation")

    monkeypatch.setattr(cli, "RunHelper", forbidden)
    with pytest.raises(Exception):
        cli.reverify_rollouts()


def test_unavailable_systemd_manager_prevents_runtime_construction(config, monkeypatch):
    from types import SimpleNamespace

    from nemo_gym.cli import service_ownership

    monkeypatch.setenv("NEMO_GYM_SERVICE_OWNERSHIP_MODE", "systemd")
    monkeypatch.setattr(service_ownership.sys, "platform", "linux")
    monkeypatch.setattr(
        service_ownership.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=1, stdout="")
    )
    monkeypatch.setattr(cli, "get_global_config_dict", lambda: config)
    monkeypatch.setattr(cli, "RunHelper", lambda: pytest.fail("runtime constructed with inaccessible user manager"))
    with pytest.raises(RuntimeError, match="reachable user manager before native startup"):
        cli.reverify_rollouts()


def test_real_runhelper_startup_failure_closes_attached_lease(config, monkeypatch):
    lease = owner(config)
    runner = cli.RunHelper()
    runner._service_ownership = lease

    def fail(_):
        raise RuntimeError("before first spawn")

    monkeypatch.setattr(runner, "_start", fail)
    with pytest.raises(RuntimeError, match="before first spawn"):
        runner.start(None)
    assert lease.lock.closed
    assert runner._service_ownership is None
