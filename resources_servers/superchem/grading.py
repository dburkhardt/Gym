# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Creator ACC extraction/score, with duplicate-choice diagnostic kept separate."""

import re


def grade(text: str, *, answer: list[str], question_type: str, **metadata: object) -> dict:
    matches = list(re.finditer(r"\\boxed\{(.+?)\}", text, re.DOTALL))
    if not matches:
        return {"reward": 0.0, "diagnostics": {"parse_error": "missing boxed answer"}}
    predicted = matches[-1].group(1).strip()
    for token in ["\\", "{", "}", "text", "math", "bf", "rm"]:
        predicted = predicted.replace(token, "")
    predicted = predicted.strip()
    if question_type == "multiple_choice":
        score = len(predicted) == len(answer) and all(a.upper() in answer for a in predicted)
        strict = set(predicted.upper()) == set(answer) and len(predicted) == len(answer)
    elif question_type == "fill_blank":
        score = predicted.lower() == answer[0].lower()
        strict = score
    else:
        raise ValueError(f"Unsupported question type: {question_type}")
    return {"reward": float(score), "diagnostics": {"predicted": predicted, "strict_set_score": float(strict)}}
