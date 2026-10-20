# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from resources_servers.moleculeqa.grading import grade


def test_controls():
    assert grade("B", answer="B")["reward"] == 1
    assert grade("A", answer="B")["reward"] == 0
    assert grade("", answer="B")["reward"] == 0
    assert grade("Answer: B", answer="B")["reward"] == 0
