# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run with the pinned evaluator's Python; no model calls. Pass scorer checkout as argv[1]."""

import sys
import tempfile
from pathlib import Path


sys.path.insert(0, sys.argv[1])
import utils.agents  # noqa: F401 - initialize upstream packages in CLI import order
from tools.fasta import FASTAHandler


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    reference = root / "reference.fa"
    reference.write_text(">seq\nACGT\n")
    handler = FASTAHandler()
    for name, value, expected in [
        ("correct", ">seq\nACGT\n", 1.0),
        ("wrong", ">seq\nAAAA\n", 0.0),
        ("malformed", "not a FASTA\n", 0.0),
        ("empty", "", 0.0),
    ]:
        candidate = root / (name + ".fa")
        candidate.write_text(value)
        result = handler.compare(str(reference), str(candidate), strategy="exact")
        assert result.similarity == expected, (name, result)
        print(name, result.status, result.similarity)
    result = handler.compare(str(reference), str(root / "missing.fa"), strategy="exact")
    assert result.status == "invalid" and result.similarity == 0.0, result
    print("missing", result.status, result.similarity)
