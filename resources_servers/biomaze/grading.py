# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TrueFalse creator parser, retaining its false-pass diagnostic."""


def grade(text: str, *, answer: str, **metadata: object) -> dict:
    normalized = text.lower().strip().strip(".")
    if normalized == "yes" or ("yes" in normalized and "guess" in normalized):
        predicted = "Yes"
    elif normalized == "no" or ("no" in normalized and "guess" in normalized):
        predicted = "No"
    else:
        predicted = "Yes" if "yes" in normalized else "No"
    return {
        "reward": float(predicted == answer),
        "diagnostics": {
            "predicted": predicted,
            "strict_valid": normalized in {"yes", "no"},
            "strict_score": float(normalized == answer.lower()),
        },
    }
