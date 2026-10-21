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


@pytest.mark.parametrize("plan_mode", ["flag", "yaml"])
def test_actual_plan_only_never_probes_ports_or_starts_services(tmp_path, plan_mode):
    import os
    import subprocess
    import sys
    from pathlib import Path

    source = Path(__file__).resolve().parents[2]
    sample = tmp_path / "sample.jsonl"
    sample.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": f"benign-{i}",
                    "responses_create_params": {"input": [{"role": "user", "content": "Choose Option 1"}]},
                    "verifier_metadata": {"expected_answer": 1, "option_count": 2},
                }
            )
            + "\n"
            for i in range(4)
        )
    )
    config = {
        "config_paths": ["benchmarks/proteinlmbench/config.yaml"],
        "policy_api_key": "offline",
        "policy_base_url": "http://127.0.0.1:1/v1",
        "use_cached_prepared_benchmarks": True,
        "proteinlmbench": {
            "resources_servers": {
                "proteinlmbench": {
                    "datasets": [
                        {
                            "name": "proteinlmbench",
                            "type": "benchmark",
                            "jsonl_fpath": str(sample),
                            "prepare_script": "benchmarks/proteinlmbench/prepare.py",
                            "num_repeats": 2,
                        }
                    ]
                }
            }
        },
    }
    if plan_mode == "yaml":
        config["plan_only"] = True
    cfg = tmp_path / "runtime.json"
    cfg.write_text(json.dumps(config))
    script = """from unittest.mock import patch
from nemo_gym.cli.main import main
with patch("nemo_gym.global_config._find_open_port_using_range", side_effect=AssertionError("port probe forbidden")), patch("nemo_gym.cli.env.RunHelper.start", side_effect=AssertionError("services forbidden")):
    main()
"""
    env = dict(os.environ, PYTHONPATH=str(source), NEMO_GYM_EXTRA_ROOTS=str(source))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "eval",
            "run",
            "--config",
            str(cfg),
            "--model-type",
            "openai_model",
            "--model",
            "fixture",
            "--split",
            "benchmark",
            "--limit",
            "4",
            *(["--plan-only"] if plan_mode == "flag" else []),
            "--output",
            str(tmp_path / "results.jsonl"),
        ],
        cwd=source,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads((tmp_path / "results_dispatch_plan.json").read_text())
    assert plan["rollouts"] == 8
    assert {row["task_id"]: row["rollouts"] for row in plan["task_distribution"]} == {
        f"benign-{i}": 2 for i in range(4)
    }
    assert not (tmp_path / "results.jsonl").exists()


def test_plan_only_no_serve_refused_before_dispatch(monkeypatch):
    from argparse import Namespace

    from nemo_gym.cli.main import _eval_run

    monkeypatch.setattr("nemo_gym.cli.main.dispatch", lambda *args: pytest.fail("must not dispatch"))
    with pytest.raises(SystemExit, match="cannot combine"):
        _eval_run(Namespace(plan_only=True, no_serve=True, target=None), ["+plan_only=true"])


@pytest.mark.parametrize("value", ["true", "false", 1, 0, None])
def test_plan_only_parser_rejects_non_boolean(value):
    from nemo_gym.config_types import ConfigError
    from nemo_gym.global_config import GlobalConfigDictParserConfig

    config = GlobalConfigDictParserConfig(
        skip_load_from_cli=True,
        skip_load_from_dotenv=True,
        initial_global_config_dict=OmegaConf.create({"plan_only": value}),
    )
    with pytest.raises(ConfigError, match="must be a boolean"):
        GlobalConfigDictParser().parse(config)


def test_cli_false_overrides_yaml_plan_only(monkeypatch):
    from nemo_gym.global_config import GlobalConfigDictParserConfig

    parser = GlobalConfigDictParser()
    monkeypatch.setattr(
        GlobalConfigDictParser,
        "parse_global_config_dict_from_cli",
        lambda self: OmegaConf.create({"plan_only": False}),
    )
    # A real component with no port ensures runtime parsing probes, rather than
    # silently inheriting offline mode from the YAML setting.
    initial = OmegaConf.create(
        {
            "plan_only": True,
            "resource": {
                "resources_servers": {"impl": {"entrypoint": "app.py", "domain": "knowledge", "verified": False}}
            },
        }
    )
    monkeypatch.setattr(
        "nemo_gym.global_config._find_open_port_using_range",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("runtime port probe")),
    )
    with pytest.raises(RuntimeError, match="runtime port probe"):
        parser.parse(GlobalConfigDictParserConfig(skip_load_from_dotenv=True, initial_global_config_dict=initial))
