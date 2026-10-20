# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from benchmarks.fukuyamabench import prepare as module


def row():
    return {
        "uuid": "A059",
        "stratum": "A",
        "extra": "preserve",
        "responses_create_params": {"input": [{"role": "user", "content": "unchanged prompt"}]},
        "verifier_metadata": {"pathway": [{"step_id": "1", "products": ["CO"]}], "checkpoints": [[1]]},
    }


def setup(monkeypatch, tmp_path, rows):
    source = tmp_path / "original.jsonl"
    source.write_text("\n".join(json.dumps(value) for value in rows) + "\n")
    monkeypatch.setenv("FUKUYAMA_FROZEN_SAMPLE", str(source))
    monkeypatch.delenv("CAMPAIGN_SAMPLE_SHA256", raising=False)
    monkeypatch.setattr(module, "OUTPUT_FPATH", tmp_path / "out" / "sample.jsonl")
    return source


def test_preservation_and_original_provenance(monkeypatch, tmp_path):
    original = row()
    source = setup(monkeypatch, tmp_path, [original])
    result = module.prepare()
    actual = json.loads(result.read_text())
    for key, value in original.items():
        if key != "verifier_metadata":
            assert actual[key] == value
    for key, value in original["verifier_metadata"].items():
        assert actual["verifier_metadata"][key] == value
    assert actual["task_id"] == "A059"
    assert actual["verifier_metadata"] | {} == original["verifier_metadata"] | {
        "gt_pathway": original["verifier_metadata"]["pathway"],
        "case_id": "A059",
        "case_set": "A",
        "lenient": True,
        "n_gt_steps": 1,
    }
    receipt = json.loads(result.with_suffix(".source.json").read_text())
    assert receipt["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt["output_sha256"] == hashlib.sha256(result.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "change",
    [
        {"pathway": []},
        {"pathway": [None]},
        {"pathway": [{"products": [None]}]},
        {"checkpoints": []},
        {"checkpoints": [[2]]},
        {"checkpoints": [[True]]},
        {"case_id": "different"},
        {"lenient": False},
    ],
)
def test_bad_metadata_not_published(monkeypatch, tmp_path, change):
    value = copy.deepcopy(row())
    value["verifier_metadata"].update(change)
    setup(monkeypatch, tmp_path, [value])
    with pytest.raises(ValueError):
        module.prepare()
    assert not module.OUTPUT_FPATH.exists()


def test_duplicate_ids(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path, [row(), row()])
    with pytest.raises(ValueError, match="Duplicate"):
        module.prepare()


def test_frozen_limit_rejected(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path, [row()])
    with pytest.raises(ValueError, match="resample"):
        module.prepare(limit=1)


def test_default_delegation(monkeypatch, tmp_path):
    monkeypatch.delenv("FUKUYAMA_FROZEN_SAMPLE", raising=False)
    monkeypatch.setattr(module, "OUTPUT_FPATH", tmp_path / "full.jsonl")
    observed = []
    monkeypatch.setattr(
        module, "_load_server_prepare", lambda: SimpleNamespace(main=lambda: observed.extend(sys.argv))
    )
    before = sys.argv
    assert module.prepare(limit=3) == module.OUTPUT_FPATH
    assert observed == [
        "prepare_fukuyamabench",
        "--output",
        str(module.OUTPUT_FPATH),
        "--sets",
        "B",
        "C",
        "--limit",
        "3",
    ]
    assert sys.argv is before
