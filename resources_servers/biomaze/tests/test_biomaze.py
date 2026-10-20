# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from resources_servers.biomaze.grading import grade


def test_controls_and_upstream_false_pass():
    assert grade("Yes", answer="Yes")["reward"] == 1
    assert grade("No", answer="Yes")["reward"] == 0
    assert grade("", answer="Yes")["reward"] == 0
    result = grade("", answer="No")
    assert result["reward"] == 1
    assert result["diagnostics"]["strict_valid"] is False
    assert grade("nonsense", answer="No")["diagnostics"]["strict_score"] == 0


def test_openended_parser():
    from resources_servers.biomaze.openended import parse_judge

    assert parse_judge("Comment: matches.\nJudge: Correct")
    assert not parse_judge("Judge: Incorrect")
    assert not parse_judge("")
    assert not parse_judge("Judge: Correct\nextra")
