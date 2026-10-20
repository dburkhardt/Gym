# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from resources_servers.fdarxbench.grading import judge_query, parse_judge


def test_rubric_parser():
    assert parse_judge("A: All required information matches.") == "A"
    assert parse_judge("B: A key dose differs.") == "B"
    assert parse_judge(" C: No answer was supplied. ") == "C"
    for text in ["", "CORRECT", "A and B", "A", "A:", "A: ", "A: correct\nB: incorrect", "A: correct B: incorrect", "D: unknown"]:
        with pytest.raises(ValueError):
            parse_judge(text)
    assert judge_query("{question}|{target}|{predicted_answer}", "q", "r", "x") == "q|r|x"
