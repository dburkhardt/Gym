# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise maintained adapters against real atomic staging and interrupted copies."""

import json
import runpy
from pathlib import Path

import pytest

from nemo_gym import dataset_staging


ADAPTERS = [
    "qcbench",
    "fukuyama",
    "smolinstruct",
    "chemiq",
    "moleculeqa",
    "scienceqa",
    "ctbench",
    "mollangbench",
    "bioplanner",
    "moleculariq",
    "molbench",
    "molinstructions",
    "scibench",
    "fgbench",
    "genomeqa",
    "csmed_ft",
    "sino_us_drugqa",
    "trialpanorama_screening",
    "s2tomg",
    "ether0_benchmark",
    "hiscibench",
    "fdarxbench",
    "suprabench",
    "lpm24",
    "omebench",
    "superchem",
    "molpuzzle",
    "evidencebench",
]


@pytest.mark.parametrize("name", ADAPTERS)
def test_preparation_preserves_existing_dataset_after_interruption(name, tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    original = repo / "benchmarks" / name / "prepare.py"
    script = tmp_path / "benchmarks" / name / "prepare.py"
    script.parent.mkdir(parents=True)
    script.write_bytes(original.read_bytes())
    source = tmp_path / "sample.jsonl"
    row = {
        "uuid": "frozen-1",
        "responses_create_params": {"input": [{"role": "user", "content": "question"}]},
        "verifier_metadata": {"answer": "answer"},
    }
    source.write_text(json.dumps(row) + "\n")
    monkeypatch.setenv("CAMPAIGN_SAMPLE_JSONL", str(source))
    runpy.run_path(str(script), run_name="__main__")
    destinations = list(tmp_path.glob("**/data/benchmark.jsonl"))
    assert len(destinations) == 1
    destination = destinations[0]
    old_bytes = destination.read_bytes()
    assert json.loads(old_bytes) == {**row, "task_id": row["uuid"]}
    assert json.loads(source.read_bytes()) == row
    receipt = json.loads(destination.with_suffix(".source.json").read_text())
    assert receipt["task_ids"] == [row["uuid"]]
    row["uuid"] = "frozen-2"
    source.write_text(json.dumps(row) + "\n")
    copy = dataset_staging.shutil.copyfileobj

    def interrupted(incoming, outgoing, length):
        outgoing.write(incoming.read(11))
        raise OSError("injected staging interruption")

    with monkeypatch.context() as scoped:
        scoped.setattr(dataset_staging.shutil, "copyfileobj", interrupted)
        with pytest.raises(OSError, match="injected staging interruption"):
            runpy.run_path(str(script), run_name="__main__")
    assert dataset_staging.shutil.copyfileobj is copy
    assert destination.read_bytes() == old_bytes
    runpy.run_path(str(script), run_name="__main__")
    assert json.loads(destination.read_bytes()) == {**row, "task_id": row["uuid"]}
    assert json.loads(source.read_bytes()) == row
    assert json.loads(destination.with_suffix(".source.json").read_text())["task_ids"] == [row["uuid"]]
