# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from benchmarks.moosechem.qualify_replay import OfflineSequence, ReplayRefused, StrictSavedClient


def record():
    return {
        "stage": "evaluate",
        "request": {"model": "fixture", "max_tokens": 16384},
        "response": {
            "id": "saved",
            "object": "chat.completion",
            "created": 1,
            "model": "fixture",
            "choices": [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": None}}],
        },
    }


def test_exact_budget_outcome_replayed_then_exhaustion_refused(tmp_path):
    path = tmp_path / "saved.jsonl"
    path.write_text(json.dumps(record()) + "\n")
    sequence = OfflineSequence(path)
    client = StrictSavedClient(
        stage="evaluate", trace=tmp_path / "trace.jsonl", replay=sequence, output_token_budget=16384
    )
    result = client.create(model="fixture", max_tokens=8192)
    assert result.choices[0].finish_reason == "length" and result.choices[0].message.content is None
    with pytest.raises(ReplayRefused, match="exhausted"):
        client.create(model="fixture", max_tokens=8192)


def test_mismatch_and_ambiguous_capture_refused(tmp_path):
    path = tmp_path / "saved.jsonl"
    row = record()
    path.write_text(json.dumps(row) + "\n")
    client = StrictSavedClient(
        stage="wrong", trace=tmp_path / "trace.jsonl", replay=OfflineSequence(path), output_token_budget=16384
    )
    with pytest.raises(ReplayRefused, match="divergence"):
        client.create(model="fixture", max_tokens=8192)
    row["error_type"] = "timeout"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ReplayRefused, match="Unresolved"):
        OfflineSequence(path)
