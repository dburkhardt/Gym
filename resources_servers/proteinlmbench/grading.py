# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original first-number accuracy, with a strict-format diagnostic."""

import re


def answer_text(text: str) -> str:
    """Ignore reasoning blocks; separately delivered reasoning is not answer text."""
    text = re.sub(r"<(think|thinking)\b[^>]*>.*?(?:</\1\s*>|$)", "", text, flags=re.I | re.S)
    return re.split(r"</(?:think|thinking)\s*>", text, flags=re.I)[-1].strip()


def grade(text: str, *, expected_answer: int, option_count: int) -> dict:
    """Score upstream first numeral and expose a stricter diagnostic extraction."""
    text = answer_text(text)
    first = re.search(r"\d+", text)
    predicted = int(first.group()) if first else None
    strict = re.fullmatch(r"(?:The correct option is\s*:?\s*)?(?:option\s*)?(\d+)[.\s]*", text, re.I)
    strict_answer = int(strict.group(1)) if strict else None
    return {
        "reward": float(predicted == expected_answer),
        "predicted_answer": predicted,
        "strict_answer": strict_answer,
        "strict_correct": strict_answer == expected_answer,
        "out_of_range": predicted is not None and not 1 <= predicted <= option_count,
    }
