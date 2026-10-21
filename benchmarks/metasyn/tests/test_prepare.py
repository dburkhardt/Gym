# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.metasyn.prepare import stage


SOURCE = (
    Path(__file__).resolve().parents[5] / "outputs/broad-eval-campaign/harbor/agent-expansion/metasyn-materialized"
)


@pytest.mark.parametrize("count", [5, 20])
def test_exact_native_prompts_and_all_fields_preserved(count, tmp_path):
    if not SOURCE.exists():
        pytest.skip("Preserved native source unavailable")
    names = ["gym-pilot5.jsonl"] + (["gym-remaining15.jsonl"] if count == 20 else [])
    original = [json.loads(line) for name in names for line in (SOURCE / name).read_text().splitlines()]
    out = stage(SOURCE, tmp_path / "benchmark.jsonl", count)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == count
    for old, new in zip(original, rows, strict=True):
        rid = old["verifier_metadata"]["review_id"]
        assert new.pop("task_id") == f"metasyn::{rid}"
        assert new == old
        assert (
            new["responses_create_params"]["input"][0]["content"]
            == (SOURCE / "rag" / str(rid) / "prompt.txt").read_text()
        )
    proof = json.loads(out.with_suffix(".source.json").read_text())
    assert proof["output_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert [r["sha256"] for r in proof["sources"]] == [
        hashlib.sha256((SOURCE / n).read_bytes()).hexdigest() for n in names
    ]
    assert len(proof["prompts"]) == count


@pytest.mark.parametrize("fault", ["prompt", "corpus_search", "missing"])
def test_changed_or_wrong_protocol_rejected(fault, tmp_path):
    if not SOURCE.exists():
        pytest.skip("Preserved native source unavailable")
    staged = tmp_path / "source"
    staged.mkdir()
    shutil.copyfile(SOURCE / "rag-preparation.json", staged / "rag-preparation.json")
    shutil.copyfile(SOURCE / "gym-pilot5.jsonl", staged / "gym-pilot5.jsonl")
    rows = [json.loads(x) for x in (SOURCE / "gym-pilot5.jsonl").read_text().splitlines()]
    for row in rows:
        rid = str(row["verifier_metadata"]["review_id"])
        target = staged / "rag" / rid / "prompt.txt"
        target.parent.mkdir(parents=True)
        shutil.copyfile(SOURCE / "rag" / rid / "prompt.txt", target)
    first = staged / "rag" / str(rows[0]["verifier_metadata"]["review_id"]) / "prompt.txt"
    if fault == "prompt":
        first.write_text("altered prompt")
    elif fault == "missing":
        first.unlink()
    else:
        shutil.copyfile(SOURCE / "agent-inputs-5.jsonl", staged / "gym-pilot5.jsonl")
    with pytest.raises((ValueError, FileNotFoundError)):
        stage(staged, tmp_path / "out.jsonl", 5)
    assert not (tmp_path / "out.jsonl").exists()


def test_unpreserved_native50_not_invented(tmp_path):
    with pytest.raises(ValueError, match="5 or20"):
        stage(tmp_path, tmp_path / "out.jsonl", 50)


def test_native_registration_preserves_runner_interface():
    root = Path(__file__).resolve().parents[3]
    cfg = OmegaConf.load(root / "responses_api_agents/metasyn_native/configs/metasyn.yaml")
    agent = cfg.metasyn_agent.responses_api_agents.metasyn_native
    assert agent.datasets[0].name == "metasyn"
    assert agent.datasets[0].prepare_script == "benchmarks/metasyn/prepare.py"
    assert agent.datasets[0].num_repeats == 1
    assert "materialized_root" in agent and "runner_script" in agent and "reference_file" in agent
