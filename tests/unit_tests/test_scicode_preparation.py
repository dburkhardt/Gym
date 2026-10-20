# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SciCode publication keeps scientific rows intact and pins their source."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from benchmarks.scicode import prepare as module


def test_pinned_preparation_preserves_scientific_content(tmp_path, monkeypatch):
    entry = {
        "problem_id": "5",
        "problem_name": "scientific example",
        "required_dependencies": ["numpy"],
        "sub_steps": [{"step_number": "1", "test_cases": ["assert result == 3"]}],
    }
    calls = []

    def load_dataset(name, **kwargs):
        calls.append((name, kwargs))
        return {"test": [entry]}

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(module, "OUTPUT_FPATH", tmp_path / "rows.jsonl")
    result = module.prepare()
    row = json.loads(result.read_text())
    assert calls == [("SciCode1/SciCode", {"revision": module.DATASET_REVISION, "token": None})]
    assert row["sub_steps"] == entry["sub_steps"]
    assert row["required_dependencies"] == entry["required_dependencies"]
    assert row["responses_create_params"] == {"input": []}
    assert row["uuid"] == row["task_id"] == "5"
    receipt = json.loads(result.with_suffix(".source.json").read_text())
    assert receipt["output_sha256"] == hashlib.sha256(result.read_bytes()).hexdigest()
    assert receipt["task_ids"] == ["5"]

    # A partial publication must retain the last complete dataset.
    original = result.read_bytes()

    def interrupted(*args, **kwargs):
        raise OSError("injected publication interruption")

    monkeypatch.setattr(module, "stage_file", interrupted)
    with pytest.raises(OSError, match="publication interruption"):
        module.prepare()
    assert result.read_bytes() == original
