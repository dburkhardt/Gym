# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from resources_servers.superchem.grading import grade


def test_controls():
    kw = dict(answer=["A"], question_type="multiple_choice")
    assert grade(r"\boxed{A}", **kw)["reward"] == 1
    assert grade(r"\boxed{B}", **kw)["reward"] == 0
    assert grade("", **kw)["reward"] == 0
    assert grade("A", **kw)["reward"] == 0
    assert grade(r"\boxed{B} then \boxed{A}", **kw)["reward"] == 1


def test_upstream_duplicate_choice_defect_preserved_and_flagged():
    result = grade(r"\boxed{AA}", answer=["A", "B"], question_type="multiple_choice")
    assert result["reward"] == 1
    assert result["diagnostics"]["strict_set_score"] == 0


def test_fill():
    assert grade(r"\boxed{Water}", answer=["water"], question_type="fill_blank")["reward"] == 1
