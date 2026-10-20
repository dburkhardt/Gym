# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content pins for the creator code used by the frozen GeneHop protocol."""

import hashlib
from pathlib import Path


CREATOR_REVISION = "0a2a327a507be49e660597769cc53431abd6613a"
SOURCE_HASHES = {
    "main_turbo.py": "cd2664646cf020d358076bf7521942eedfef02b42f7944973df3d1151bca5817",
    "evaluate.py": "1a3d959f7ba98d9a4eb13d6c34aad9a060ab2f43549f413f73b0f156526db86d",
}


def validate_source(root: Path) -> dict[str, str]:
    """Fail before inference if either executed creator source differs."""
    observed = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_HASHES}
    if observed != SOURCE_HASHES:
        raise ValueError("GeneHop creator prompt/scorer source differs from pinned revision")
    return observed
