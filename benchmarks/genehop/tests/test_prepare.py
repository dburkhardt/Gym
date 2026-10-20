# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from benchmarks.genehop.prepare import prepare


def test_native_prepare_preserves_frozen_bytes(tmp_path, monkeypatch):
    path = tmp_path / "sample.jsonl"
    row = {
        "responses_create_params": {"input": [{"role": "user", "content": "question"}]},
        "verifier_metadata": {"question": "question", "answer": "hidden reference"},
    }
    original = (json.dumps(row) + "\n").encode()
    path.write_bytes(original)
    monkeypatch.setenv("GENEHOP_SAMPLE", str(path))
    assert prepare() == path
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "content", ["", '{"responses_create_params":{"input":[]},"verifier_metadata":{"question":"question"}}']
)
def test_native_prepare_rejects_invalid_sample(tmp_path, monkeypatch, content):
    path = tmp_path / "sample.jsonl"
    path.write_text(content)
    monkeypatch.setenv("GENEHOP_SAMPLE", str(path))
    with pytest.raises(ValueError):
        prepare()
