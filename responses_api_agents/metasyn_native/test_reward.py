# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from app import native_reward


@pytest.mark.parametrize("metric", [None, True, "0.8", float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_metric_is_not_model_zero(metric):
    with pytest.raises(ValueError):
        native_reward({"status": "succeeded", "metrics": {"inclusion_f1": metric}})


@pytest.mark.parametrize("status", ["failed", "partial", None])
def test_failed_grading_is_not_model_score(status):
    with pytest.raises(ValueError):
        native_reward({"status": status, "metrics": {"inclusion_f1": 0.8}})


def test_native_zero_and_nonzero_are_preserved():
    for score in [0, 0.8, 1]:
        assert native_reward({"status": "succeeded", "metrics": {"inclusion_f1": score}}) == score
