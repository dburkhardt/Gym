# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib

import pytest

from benchmarks.genehop import source_identity


def test_creator_source_mutation_is_rejected(tmp_path, monkeypatch):
    expected = {}
    for name in source_identity.SOURCE_HASHES:
        data = ("original " + name).encode()
        (tmp_path / name).write_bytes(data)
        expected[name] = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(source_identity, "SOURCE_HASHES", expected)
    assert source_identity.validate_source(tmp_path) == expected
    (tmp_path / "evaluate.py").write_text("changed scorer")
    with pytest.raises(ValueError, match="pinned revision"):
        source_identity.validate_source(tmp_path)
