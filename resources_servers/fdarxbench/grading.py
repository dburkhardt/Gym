# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FDA SimpleQA rubric formatting; no answer matching substituted for judging."""

import re


def judge_query(template: str, question: str, reference: str, candidate: str) -> str:
    return template.format(question=question, target=reference, predicted_answer=candidate)


def parse_judge(text: str) -> str:
    # Creator rubric explicitly requests "LETTER: short reason" on one line.
    match = re.fullmatch(r"([ABC]):[^\S\r\n]+([^\r\n]+)", text.strip())
    if match is None or not match.group(2).strip():
        raise ValueError("Judge did not return creator LETTER: short reason format")
    # Reject another verdict masquerading as part of a contradictory rationale.
    if re.search(r"(?:^|\s)[ABC]:", match.group(2)):
        raise ValueError("Judge returned multiple verdict labels")
    return match.group(1)
