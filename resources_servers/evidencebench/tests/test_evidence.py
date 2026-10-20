# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from resources_servers.evidencebench.grading import grade


def test_controls():
    meta = dict(sentence_index2aspects={"0": ["a"], "1": ["b"], "2": []}, limit=2, candidate_count=3)
    assert grade("DECISION: [0, 1]", **meta)["reward"] == 1
    assert grade("DECISION: [2]", **meta)["reward"] == 0
    assert grade("", **meta)["reward"] == 0
    assert grade("DECISION: [x]", **meta)["reward"] == 0
    assert grade("DECISION: [1, 0]", **meta)["reward"] == 1
    assert abs(grade("DECISION: [0, 1, 2]", **meta)["reward"] - 2 / 3) < 1e-12
