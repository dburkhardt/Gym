# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from resources_servers.microvqa.grading import grade


def test_controls():
    assert grade("The answer is (2)", correct_index=1)["reward"] == 1
    assert grade("The answer is (1)", correct_index=1)["reward"] == 0
    assert grade("", correct_index=1)["reward"] == 0
    assert grade("2", correct_index=1)["reward"] == 0
    assert grade("The answer is **(2)**", correct_index=1)["reward"] == 1
    assert grade("The answer is (2), the answer is (3)", correct_index=1)["reward"] == 1
