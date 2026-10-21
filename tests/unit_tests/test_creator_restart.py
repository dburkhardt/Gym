# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from nemo_gym.config_types import ConfigError
from nemo_gym.creator_restart import creator_source_binding, exact_creator_graph


def graph():
    return {
        "policy": {"responses_api_models": {"openai_model": {}}},
        "solver": {
            "responses_api_agents": {
                "simple_agent": {
                    "model_server": {"name": "policy"},
                    "resources_server": {"name": "science"},
                    "max_steps": 1,
                }
            }
        },
        "science": {"resources_servers": {"campaign_creator_scoring": {"benchmark": "genomeqa"}}},
        "relay": {"environment_servers": {"legacy_agent": {"agent_server": {"name": "solver"}}}},
    }


@pytest.mark.parametrize("change", ["additional", "provider", "tools", "mcp", "steps", "redirect", "entrypoint"])
def test_uninspected_graph_refused(change):
    config = graph()
    assert exact_creator_graph(config)["resource"]["settings"]["benchmark"] == "genomeqa"
    agent = config["solver"]["responses_api_agents"]["simple_agent"]
    if change == "additional":
        config["other"] = {"resources_servers": {"mcqa": {}}}
    elif change == "provider":
        config["remote"] = {"sandbox_providers": {"external": {}}}
    elif change == "tools":
        agent["tools"] = [{"name": "shell"}]
    elif change == "mcp":
        agent["mcp_servers"] = ["remote"]
    elif change == "steps":
        agent["max_steps"] = 2
    elif change == "redirect":
        agent["resources_server"] = {"name": "other"}
    else:
        agent["entrypoint"] = "other.py"
    with pytest.raises(ConfigError):
        exact_creator_graph(config)


@pytest.fixture
def source(tmp_path, monkeypatch):
    from benchmarks.scientific_agent_campaign.deterministic import PINS

    repo = tmp_path / "creator"
    repo.mkdir()
    (repo / "run.py").write_text("# benign source fixture\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "run.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    pin = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setitem(PINS, "genomeqa", pin)
    root = tmp_path / "gym"
    package = root / "benchmarks/scientific_agent_campaign"
    package.mkdir(parents=True)
    for name in ("deterministic.py", "prepare.py", "recovered_releases.py"):
        (package / name).write_text("# maintained fixture\n")
    return root, {"benchmark": "genomeqa", "upstream_repo": str(repo)}


def test_source_and_maintained_adapter_bound(source):
    root, resource = source
    first = creator_source_binding(root, resource)
    assert first["creator"]["source"] == "run.py"
    helper = root / "benchmarks/scientific_agent_campaign/recovered_releases.py"
    helper.write_text("# changed adapter\n")
    assert creator_source_binding(root, resource) != first


@pytest.mark.parametrize("change", ["dirty", "symlink", "missing", "branch"])
def test_changed_assets_and_unsupported_branches_refused(source, change):
    root, resource = deepcopy(source)
    file = Path(resource["upstream_repo"]) / "run.py"
    if change == "dirty":
        file.write_text("# edited creator semantics\n")
    elif change == "symlink":
        file.unlink()
        file.symlink_to(root / "benchmarks/scientific_agent_campaign/prepare.py")
    elif change == "missing":
        file.unlink()
    else:
        resource["benchmark"] = "moleculariq"
    with pytest.raises(ConfigError):
        creator_source_binding(root, resource)


def test_prior_batch_cannot_restart_after_creator_asset_change(source, tmp_path, monkeypatch):
    from types import SimpleNamespace

    import nemo_gym.creator_restart as module
    from nemo_gym.batch_restart import bind_execution

    root, resource = source
    config = graph()
    config["science"]["resources_servers"]["campaign_creator_scoring"] = resource
    # Exercise the actual binder's capture/protocol/source comparison with the
    # real pinned-file qualification; runtime probing is covered by native CLI.
    monkeypatch.setattr(module, "qualify_creator_dependencies", lambda *_: creator_source_binding(root, resource))
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text('{"uuid":"one","responses_create_params":{"input":[]},"verifier_metadata":{"answer":"A"}}\n')
    tracker = SimpleNamespace(manifest_sha256="m" * 64, execution_binding=None)
    bind_execution(tracker, config, inputs, resume=False, require_prior=False, capture_dirs=[tmp_path / "captures"])
    prior = deepcopy(tracker.execution_binding)
    (Path(resource["upstream_repo"]) / "run.py").write_text("# changed scorer\n")
    with pytest.raises(ConfigError, match="dirty"):
        bind_execution(tracker, config, inputs, resume=True, require_prior=True, capture_dirs=[tmp_path / "captures"])
    assert tracker.execution_binding == prior
    assert not list((tmp_path / "captures").glob("*.jsonl"))
