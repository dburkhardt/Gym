# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise validity metadata against the actual pinned creator numeric scorer."""

import os
from pathlib import Path

import pytest

from benchmarks.scientific_agent_campaign.deterministic import score


@pytest.mark.parametrize(
    "reference,answer,reward,valid",
    [("3.51e-09", r"\boxed{3.51e-09}", 0, False), ("1.25", r"\boxed{1.25}", 1, True), ("1.25", r"\boxed{9}", 0, True)],
)
def test_creator_defect_is_distinct_from_model_error(reference, answer, reward, valid):
    source = os.environ.get("QCBENCH_PINNED_SOURCE")
    if not source:
        pytest.skip("Requires pinned QCBench creator checkout")
    result = score("qcbench", Path(source), {"verifier_metadata": {"answer": reference}}, answer)
    assert result["reward"] == reward
    assert result["measurement_valid"] is valid
    assert bool(result["measurement_invalid_reason"]) is (not valid)
