# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from responses_api_agents.bixbench3.runtime import check_gold_isolation, read_artifact_score


def test_gold_outside_mounts(tmp_path):
    gold = tmp_path / "host_gold"
    data = tmp_path / "input"
    gold.mkdir()
    data.mkdir()
    check_gold_isolation(gold, [data])


def test_gold_inside_mount_rejected(tmp_path):
    gold = tmp_path / "gold"
    gold.mkdir()
    with pytest.raises(ValueError, match="overlaps"):
        check_gold_isolation(gold, [tmp_path])


def test_symlink_mount_rejected(tmp_path):
    gold = tmp_path / "gold"
    gold.mkdir()
    link = tmp_path / "input"
    link.symlink_to(gold, target_is_directory=True)
    with pytest.raises(ValueError, match="overlaps"):
        check_gold_isolation(gold, [link])


@pytest.mark.parametrize("score", [0, 0.5, 1])
def test_artifact_score(tmp_path, score):
    path = tmp_path / "grade.json"
    path.write_text(json.dumps({"score": score, "artifact_count": 1, "artifacts": [{"score": score}]}))
    assert read_artifact_score(path) == score


@pytest.mark.parametrize(
    "payload",
    [
        {"score": 1},  # Inspect completeness is not an artifact score.
        {"score": float("nan"), "artifact_count": 1, "artifacts": [{}]},
        {"score": True, "artifact_count": 1, "artifacts": [{}]},
        {"score": 2, "artifact_count": 1, "artifacts": [{}]},
        {"score": 1, "artifact_count": 2, "artifacts": [{}]},
        {"score": 0, "artifact_count": 0, "artifacts": []},
    ],
)
def test_invalid_score_rejected(tmp_path, payload):
    path = tmp_path / "grade.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        read_artifact_score(path)
