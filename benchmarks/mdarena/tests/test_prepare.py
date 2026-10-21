# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.mdarena.prepare import stage


SAMPLES = Path(__file__).resolve().parents[5] / "outputs/broad-eval-campaign/harbor/mdarena"


@pytest.mark.parametrize("count", [5, 20, 50])
def test_frozen_rows_preserved_and_original_checksum(count, tmp_path):
    source = SAMPLES / f"pilot-{count}.jsonl"
    if not source.exists():
        pytest.skip("Preserved campaign sample not available")
    original = [json.loads(line) for line in source.read_text().splitlines()]
    out = stage(source, tmp_path / "benchmark.jsonl")
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == count
    for old, new in zip(original, rows, strict=True):
        assert new.pop("task_id") == old["instance_id"]
        assert new == old
    proof = json.loads(out.with_suffix(".source.json").read_text())
    assert proof["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert proof["output_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert proof["task_ids"] == [x["instance_id"] for x in original]
    assert not proof["scientific_task_files_modified"]


@pytest.mark.parametrize("change", ["duplicate", "reorder", "temperature", "truncate"])
def test_altered_frozen_source_rejected(change, tmp_path):
    source = SAMPLES / "pilot-5.jsonl"
    if not source.exists():
        pytest.skip("Preserved campaign sample not available")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    if change == "duplicate":
        rows[-1] = rows[0]
    elif change == "reorder":
        rows.reverse()
    elif change == "temperature":
        rows[0]["responses_create_params"]["temperature"] = 1
    else:
        rows.pop()
    changed = tmp_path / "changed.jsonl"
    changed.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    with pytest.raises(ValueError, match="frozen"):
        stage(changed, tmp_path / "out.jsonl")
    assert not (tmp_path / "out.jsonl").exists()


def test_both_harnesses_bind_same_dataset_without_changing_protocol():
    root = Path(__file__).resolve().parents[3]
    for variant in ("json", "xml"):
        cfg = OmegaConf.load(root / f"responses_api_agents/harbor_agent/configs/harbor_agent_mdarena_{variant}.yaml")
        agent = cfg.harbor_agent.responses_api_agents.harbor_agent
        assert agent.harbor_agent_kwargs.parser_name == variant
        assert agent.datasets[0].name == f"mdarena_{variant}"
        assert agent.datasets[0].prepare_script == "benchmarks/mdarena/prepare.py"
        assert agent.datasets[0].jsonl_fpath == "benchmarks/mdarena/data/benchmark.jsonl"
        assert agent.datasets[0].num_repeats == 1
        assert cfg.harbor_agent_environment_server.environment_servers.legacy_agent.agent_server.name == "harbor_agent"
