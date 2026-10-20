# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Creator first answer-is regex with zero-index gold labels."""

import re


def grade(text: str, *, correct_index: int, **metadata: object) -> dict:
    match = re.search(r"answer is \*?\*?\(?([0-9])\)?\*?\*?", text)
    predicted = int(match.group(1)) - 1 if match else -1
    return {
        "reward": float(predicted == correct_index),
        "diagnostics": {"predicted_index": predicted, "parse_ok": match is not None},
    }
