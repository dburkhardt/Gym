# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Creator conclusion-judge parser and prompt construction."""


def parse_judge(response: str) -> bool:
    if "Judge:" not in response:
        lower = response.lower()
        return any(x in lower for x in [" correct", ' "correct', " 'correct"]) and not any(
            x in lower for x in [" incorrect", ' "incorrect', " 'incorrect"]
        )
    return response.split("Judge:", 1)[1].strip().lower() == "correct"


def judge_query(template: str, question: str, reference: str, candidate: str) -> str:
    return (
        template.strip()
        + "\n\nQuestion: "
        + question.strip()
        + "\nStandard Answer: "
        + reference.strip()
        + "\nNew Answer: "
        + candidate.strip()
        + "\nComment: "
    )
