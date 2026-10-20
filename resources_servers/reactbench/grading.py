# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conservative exact-match variant of the paper's unreleased template extractor."""


def grade(text: str, *, answer: str, **metadata: object) -> dict:
    predicted = text.strip()
    return {
        "reward": float(predicted == answer),
        "diagnostics": {
            "predicted": predicted,
            "exact_bytes_score": float(text == answer),
            "protocol": "released inline prompt / outer-whitespace-trim exact-match variant",
        },
    }
