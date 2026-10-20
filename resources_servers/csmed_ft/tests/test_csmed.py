# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from resources_servers.csmed_ft.grading import grade, macro_metrics


def test_controls():
    assert grade("Included", answer="included")["reward"] == 1
    assert grade("Excluded", answer="included")["reward"] == 0
    assert grade("", answer="excluded")["reward"] == 0
    assert grade("maybe", answer="excluded")["reward"] == 0
    assert macro_metrics(["included", "excluded"], ["included", "excluded"])["macro_f1"] == 1
    assert macro_metrics(["included", "excluded"], ["excluded", "included"])["macro_f1"] == 0
