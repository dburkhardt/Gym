# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit pilot strict-letter accuracy; creator standalone parser unavailable."""

import re


def grade(text: str, *, answer: str, **metadata: object) -> dict:
    match = re.fullmatch(r"\s*([A-D])\s*[.]?\s*", text)
    predicted = match.group(1) if match else None
    return {
        "reward": float(predicted == answer),
        "diagnostics": {"predicted": predicted, "protocol": "strict-letter pilot variant"},
    }
