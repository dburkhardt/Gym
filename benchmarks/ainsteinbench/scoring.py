# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify creator raw percentages without replacing their scientific metric."""

import re


EXPECTED_TESTS = {"MSB_pyscf_pyscf_pr2373": 4, "MSB_pyscf_pyscf_pr2396": 9}


def qualification(task_id, parsed, text, return_code):
    total = parsed["passed_count"] + parsed["failed_count"]
    result = {
        "expected_control_test_count": EXPECTED_TESTS[task_id],
        "observed_test_count": total,
        "raw_metric": "creator percent passed; includes already-passing baseline tests, not solved-task rate",
    }
    if return_code < 0 or return_code >= 126:
        return {
            **result,
            "valid": False,
            "kind": "infrastructure_execution_failure",
            "reason": "signal termination or unavailable executable",
        }
    patch_rejection = (
        return_code == 1
        and "Error: git apply failed" in text
        and re.search(
            r"(?m)^error: (?:No valid patches in input|corrupt patch at line|patch failed:|.*patch does not apply|.*does not exist in index)",
            text,
        )
    )
    if total == 0 and patch_rejection:
        return {
            **result,
            "valid": True,
            "kind": "model_patch_rejected",
            "reason": "creator git apply explicitly rejected submitted patch before tests",
        }
    if total == 0:
        return {
            **result,
            "valid": False,
            "kind": "unclassified_no_tests",
            "reason": "no parsed tests without a recognized creator patch-rejection diagnostic",
        }
    if total < EXPECTED_TESTS[task_id] or "ERROR collecting" in text or "INTERNALERROR" in text:
        return {
            **result,
            "valid": False,
            "kind": "incomplete_or_failed_test_collection",
            "reason": "raw percentage retained, but expected control test coverage was not established",
        }
    return {
        **result,
        "valid": True,
        "kind": "creator_tests_scored",
        "reason": "raw creator test percentage; additional tests, if present, remain explicitly counted",
    }
