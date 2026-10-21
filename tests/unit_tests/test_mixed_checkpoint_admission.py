# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent scientific-route rejection controls, without inference."""

from copy import deepcopy

import pytest

from nemo_gym.creator_restart import creator_graph_for_row, creator_source_binding, exact_creator_graphs
from nemo_gym.global_config import ConfigError


def mixed_graph():
    config = {"policy": {"responses_api_models": {"openai_model": {}}}}
    for name in ("qcbench", "moljson", "genomeqa"):
        config[name] = {"resources_servers": {"campaign_creator_scoring": {"benchmark": name}}}
        config[name + "_agent"] = {
            "responses_api_agents": {
                "simple_agent": {
                    "max_steps": 1,
                    "model_server": {"type": "responses_api_models", "name": "policy"},
                    "resources_server": {"type": "resources_servers", "name": name},
                }
            }
        }
        config[name + "_relay"] = {
            "environment_servers": {
                "legacy_agent": {"agent_server": {"type": "responses_api_agents", "name": name + "_agent"}}
            }
        }
    return config


def routed_row():
    return {
        "agent_ref": {"name": "qcbench_agent"},
        "_ng_environment_server": "qcbench_relay",
        "task_source": "qcbench",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("agent_ref", {"name": "moljson_agent"}),
        ("_ng_environment_server", "moljson_relay"),
        ("task_source", "moljson"),
        ("task_source", None),
    ],
)
def test_cross_wired_scientific_route_rejected(field, value):
    graphs = exact_creator_graphs(mixed_graph())
    row = routed_row()
    assert creator_graph_for_row(graphs, row)["resource"]["name"] == "qcbench"
    row[field] = value
    before = deepcopy(row)
    with pytest.raises(ConfigError, match="cross-wired|selected scorer"):
        creator_graph_for_row(graphs, row)
    assert row == before


def test_missing_selected_scorer_rejected():
    config = mixed_graph()
    del config["moljson"]
    with pytest.raises(ConfigError, match="outside inspected graph"):
        exact_creator_graphs(config)


@pytest.mark.parametrize("benchmark", ["qcbench", "moljson", "genomeqa"])
def test_missing_exact_creator_source_not_replaced(tmp_path, benchmark):
    absent = tmp_path / "absent"
    with pytest.raises(ConfigError):
        creator_source_binding(tmp_path, {"benchmark": benchmark, "upstream_repo": str(absent)})
    assert not absent.exists()


def test_unequal_disjoint_scopes_preserve_routed_attempts():
    from nemo_gym._checkpoint.coordination import Participant, Participants
    from nemo_gym._checkpoint.evaluation import routed_commit_scope

    participants = Participants(None, "fixture", (Participant("q", "agent"), Participant("m", "agent")))
    active = {"q1-a2": "q", "q2": "q", "m1": "m"}
    replies = {"q": {"episode_ids": ["q1-a2", "q2"]}, "m": {"episode_ids": ["m1"]}}
    assert routed_commit_scope(participants, replies, active) == set(active)
    before = deepcopy(replies)
    replies["q"]["episode_ids"] = ["q1-a2", "m1"]
    with pytest.raises(ConfigError):
        routed_commit_scope(participants, replies, active)
    assert active == {"q1-a2": "q", "q2": "q", "m1": "m"}
    replies = before
    replies["q"]["episode_ids"] = ["q1", "q2"]
    with pytest.raises(ConfigError):
        routed_commit_scope(participants, replies, active)


@pytest.mark.asyncio
async def test_finalizer_during_commit_uses_original_route_scope(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from nemo_gym._checkpoint import coordination, evaluation
    from nemo_gym.episode_types import EpisodeId

    controller = evaluation.EvaluationCheckpoint(
        directory=tmp_path / "checkpoint", client=None, config=mixed_graph(), binding={}, capture_dirs=[], artifacts=[]
    )
    controller.participants = coordination.Participants(
        None, "fixture", (coordination.Participant("qcbench_agent", "agent"),)
    )
    controller.discover = AsyncMock()
    controller.active = {"q1": EpisodeId(rollout_id="q1"), "q2": EpisodeId(rollout_id="q2")}
    controller.active_agents = {"q1": "qcbench_agent", "q2": "qcbench_agent"}
    monkeypatch.setattr(coordination, "prepare", AsyncMock(return_value=coordination.PrepareResult(True, {})))
    monkeypatch.setattr(coordination, "resume", AsyncMock())
    monkeypatch.setattr(evaluation, "capture_snapshot", lambda *args: {})
    published = []
    monkeypatch.setattr(evaluation, "publish", lambda path, payload: published.append(payload))

    async def commit(*args, **kwargs):
        # Controller's normal finalizer can settle a reply while commit is awaited.
        controller.after_persist({"_ng_rollout_id": "q1"})
        return {"qcbench_agent": {"episode_ids": ["q1", "q2"]}}

    monkeypatch.setattr(coordination, "commit", commit)
    await controller.checkpoint()
    assert len(published) == 1
    assert controller.persisted == {"q1"}
    assert controller.admission.is_set()
