# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual collator -> collector controls; no services or inference."""

import json
from collections import Counter

import pytest
from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from nemo_gym.train_data_utils import TrainDataProcessor


def collate(tmp_path, repeats=(8,), defer=True, conflict=False):
    datasets = []
    for i, repeat in enumerate(repeats):
        path = tmp_path / f"source{i}.jsonl"
        rows = [
            {
                "task_id": f"{i}:{j}",
                "responses_create_params": {"input": [{"role": "user", "content": f"benign {i}:{j}"}]},
            }
            for j in range(4)
        ]
        if conflict:
            rows[0]["_ng_dataset_num_repeats"] = 2
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        datasets.append(
            {
                "name": f"d{i}",
                "type": "benchmark",
                "jsonl_fpath": str(path),
                "num_repeats": repeat,
                "prepare_script": "unused.py",
            }
        )
    cfg = OmegaConf.create(
        {"agent": {"responses_api_agents": {"simple_agent": {"entrypoint": "app.py", "datasets": datasets}}}}
    )
    instances = GlobalConfigDictParser().filter_for_server_instance_configs(cfg)
    paths = TrainDataProcessor()._collate_samples_single_type(
        "benchmark", instances, task_data_validation="off", defer_dataset_repeats=defer
    )
    result = tmp_path / "collated.jsonl"
    result.write_text("".join(path.read_text() for path in paths))
    return result


def collect(path, **kwargs):
    cfg = RolloutCollectionConfig(
        input_jsonl_fpath=str(path), output_jsonl_fpath=str(path.parent / "out.jsonl"), **kwargs
    )
    return RolloutCollectionHelper()._preprocess_rows_from_config(cfg)


def test_four_tasks_eight_draws_not_first_task_32(tmp_path):
    rows = collect(collate(tmp_path), limit=4, num_repeats=8)
    assert Counter(row["task_id"] for row in rows) == {f"0:{j}": 8 for j in range(4)}
    assert len({(row["_ng_task_index"], row["_ng_rollout_index"]) for row in rows}) == 32
    assert all("_ng_dataset_num_repeats" not in row for row in rows)


@pytest.mark.parametrize(
    "override,expected", [({}, (2, 3)), ({"num_repeats": 1}, (1, 1)), ({"num_repeats": {"agent": 4}}, (4, 4))]
)
def test_same_agent_different_dataset_defaults(tmp_path, override, expected):
    rows = collect(collate(tmp_path, (2, 3)), **override)
    assert Counter(row["task_id"].split(":")[0] for row in rows) == {"0": 4 * expected[0], "1": 4 * expected[1]}


def test_limit_selects_source_tasks_before_repeats(tmp_path):
    rows = collect(collate(tmp_path, (2, 3)), limit=5)
    assert Counter(row["task_id"] for row in rows) == {"0:0": 2, "0:1": 2, "0:2": 2, "0:3": 2, "1:0": 3}


def test_standalone_collate_preserves_existing_expansion(tmp_path):
    path = collate(tmp_path, (2,), defer=False)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 8
    assert all("_ng_dataset_num_repeats" not in row for row in rows)


def test_reserved_source_metadata_rejected(tmp_path):
    with pytest.raises(ValueError, match="reserved"):
        collate(tmp_path, conflict=True)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8"])
def test_invalid_deferred_repeat_rejected(tmp_path, value):
    path = collate(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["_ng_dataset_num_repeats"] = value
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="positive integer"):
        collect(path)
