# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from benchmarks.csmed_ft.creator_protocol import SOURCE_SHA256, ChunkPrediction, merge_chunks, parse_label


def creator_function(filename, name):
    root = Path(__file__).resolve().parents[4] / "csmed-baselines-upstream/experiments/csmed_ft"
    path = root / filename
    if not path.exists():
        pytest.skip("Pinned creator checkout is not installed")
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256[filename]
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    scope = {"copy": copy}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


@pytest.mark.parametrize(
    "answer", ["Included", "INCLUDE", "yes", "Excluded", "EXCLUDE", "no", "", " Included", "yes\n", "unknown"]
)
def test_parser_exact_creator(answer):
    parser = creator_function("full_text_prompting.py", "parse_response")
    response = {"choices": [SimpleNamespace(message=SimpleNamespace(content=answer))]}
    expected = parser(response, {"review_id": "r", "pmid": "p", "label": 1}, 0, 0)
    assert parse_label(answer) == expected["prediction"]


def test_merge_exact_creator_and_exposes_invalid_reference():
    chunks = [
        ChunkPrediction(2, -1, 0),
        ChunkPrediction(1, 1, 1),
        ChunkPrediction(2, -1, 1),
        ChunkPrediction(1, 0, 1),
        ChunkPrediction(3, -1, 1),
        ChunkPrediction(3, 1, 1),
    ]
    original = creator_function("analyse_results.py", "merge_predictions")
    expected = original(pd.DataFrame([vars(chunk) for chunk in chunks])).to_dict("records")
    actual = merge_chunks(chunks)
    assert [{key: getattr(row, key) for key in ("uuid", "prediction", "gold_label")} for row in actual] == expected
    assert [row.prediction for row in actual] == [0, 1, 1]
    assert actual[1].invalid_chunk_count == 2
    assert actual[1].conflicting_gold_labels is True
    assert actual[1].gold_label == 0  # Preserve creator first-gold behavior, not majority vote.


def test_contract_does_not_accept_unbound_or_invalid_records():
    assert merge_chunks([]) == []
    with pytest.raises(ValueError, match="integer row"):
        merge_chunks([ChunkPrediction("1", 1, 0)])
    with pytest.raises(ValueError, match="prediction"):
        merge_chunks([ChunkPrediction(1, 2, 0)])
