# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Creator aspect recall; exact expectation replaces stochastic oversize truncation."""

import math
import re


def grade(text: str, *, sentence_index2aspects: dict, limit: int, candidate_count: int, **metadata: object) -> dict:
    normalized = text
    for token in ["```python", "```", "*", "index"]:
        normalized = normalized.replace(token, "")
    matches = re.findall(r"DECISION:\s*[-*]*\s*\[([0-9, ]+)\]", normalized)
    try:
        picked = [int(x.strip()) for x in matches[-1].split(",")] if matches else []
    except ValueError:
        picked = []
    universe = set(a for values in sentence_index2aspects.values() for a in values)
    if not universe:
        raise ValueError("Reference has no evidence aspects")
    invalid = [i for i in picked if not 0 <= i < candidate_count]
    if len(picked) <= limit:
        covered = set(a for i in picked for a in sentence_index2aspects.get(str(i), []))
        reward = len(covered) / len(universe)
    else:
        # Average over every uniformly selected size-limit subset of positions.
        n = len(picked)
        reward = sum(
            1
            - (
                math.comb(n - sum(a in sentence_index2aspects.get(str(i), []) for i in picked), limit)
                / math.comb(n, limit)
                if n - sum(a in sentence_index2aspects.get(str(i), []) for i in picked) >= limit
                else 0
            )
            for a in universe
        ) / len(universe)
    return {
        "reward": reward,
        "diagnostics": {
            "selected": picked,
            "invalid_indices": invalid,
            "over_limit": len(picked) > limit,
            "protocol": "original ER@10; exact oversize expectation",
        },
    }
