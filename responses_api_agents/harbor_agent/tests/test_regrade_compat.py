# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from responses_api_agents.harbor_agent.regrade_compat import install_regrade_compat


def test_native_replay_preserves_dangling_and_host_links(tmp_path, monkeypatch):
    from harbor.trial.regrade import RegradeTrial

    original = RegradeTrial._seed_from_source
    monkeypatch.setattr(RegradeTrial, "_seed_from_source", original)
    install_regrade_compat()
    install_regrade_compat()  # Idempotent within a native worker.
    source = tmp_path / "source"
    target = tmp_path / "target"
    workspace = source / "artifacts" / "workspace"
    workspace.mkdir(parents=True)
    (source / "artifacts" / "manifest.json").write_text("[]")
    host_secret = tmp_path / "controller-only"
    host_secret.write_text("must not be copied as file content")
    (workspace / "BASIS_MOLOPT").symlink_to("/opt/cp2k/data/BASIS_MOLOPT")
    (workspace / "host-link").symlink_to(host_secret)
    (workspace / "result.out").write_text("saved scientific output")
    trial = RegradeTrial.__new__(RegradeTrial)
    trial._source_paths = SimpleNamespace(
        agent_dir=source / "agent", artifacts_dir=source / "artifacts", trial_dir=source
    )
    trial.paths = SimpleNamespace(agent_dir=target / "agent", artifacts_dir=target / "artifacts")
    trial.agent_env_paths = SimpleNamespace(artifacts_dir=tmp_path / "convention")
    trial.task = SimpleNamespace(config=SimpleNamespace(artifacts=[]))
    trial.config = SimpleNamespace(artifacts=[])
    trial._seed_from_source()
    restored = target / "artifacts" / "workspace"
    assert (restored / "BASIS_MOLOPT").is_symlink()
    assert str((restored / "BASIS_MOLOPT").readlink()) == "/opt/cp2k/data/BASIS_MOLOPT"
    assert (restored / "host-link").is_symlink()
    assert (restored / "host-link").readlink() == host_secret
    assert (restored / "result.out").read_text() == "saved scientific output"
    assert host_secret.read_text() == "must not be copied as file content"


def test_upstream_drift_refused(monkeypatch):
    from harbor.trial.regrade import RegradeTrial

    def changed(self):
        raise AssertionError("never execute changed code")

    monkeypatch.setattr(RegradeTrial, "_seed_from_source", changed)
    with pytest.raises(RuntimeError, match="implementation changed"):
        install_regrade_compat()
