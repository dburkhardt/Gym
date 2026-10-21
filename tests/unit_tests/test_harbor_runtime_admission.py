# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse

import pytest

from nemo_gym.tasks.harbor import cli, materialize
from nemo_gym.tasks.harbor.task import HarborTaskError


def test_missing_runtime_refuses_before_fetch_or_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(materialize, "_resolve_under_cwd_or_install", lambda path: tmp_path / "missing")
    monkeypatch.setattr(cli, "prepare_target", lambda *args: pytest.fail("must not fetch or publish"))
    args = argparse.Namespace(
        target="harbor:fixture", agent="hermes", sandbox=None, no_serve=False, _parser=argparse.ArgumentParser()
    )
    with pytest.raises(SystemExit) as exc:
        cli.run_target(args, [])
    assert exc.value.code == 2
    assert "not interchangeable" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_direct_config_requires_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(materialize, "_resolve_under_cwd_or_install", lambda path: tmp_path / "missing")
    with pytest.raises(HarborTaskError, match="not runnable"):
        materialize.run_config(
            taskset="x",
            folder=tmp_path,
            tasks=[],
            rows_path=tmp_path / "rows.jsonl",
            agent_instance_name="a",
            agent_impl="a",
        )
    assert not (tmp_path / "rows.jsonl").exists()


def test_plugin_runtime_resolves_through_actual_search_roots(tmp_path, monkeypatch):
    plugin = tmp_path / "plugin"
    implementation = plugin / "resources_servers/harbor"
    implementation.mkdir(parents=True)
    (implementation / "app.py").write_text("# presence fixture, not runtime qualification\n")
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(plugin))
    assert materialize.require_resources_implementation() == implementation


def test_malformed_runtime_entrypoint_directory_refused(tmp_path, monkeypatch):
    implementation = tmp_path / "resources_servers/harbor"
    (implementation / "app.py").mkdir(parents=True)
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(tmp_path))
    with pytest.raises(HarborTaskError, match="execution is unavailable"):
        materialize.require_resources_implementation()
