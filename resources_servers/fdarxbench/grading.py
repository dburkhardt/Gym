# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FDA SimpleQA rubric formatting; no answer matching substituted for judging."""


def judge_query(template: str, question: str, reference: str, candidate: str) -> str:
    return template.format(question=question, target=reference, predicted_answer=candidate)


def parse_judge(text: str) -> str:
    label = text.strip()
    if label not in {"A", "B", "C"}:
        raise ValueError("Judge did not return required single A/B/C label")
    return label
