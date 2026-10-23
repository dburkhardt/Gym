# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retain ET creator scores without treating insensitive tests as scientific validation."""

from pathlib import Path


TASK = "ainsteinbench::ET_EinsteinInitialData_Exact_metrics_de_Sitter_F"


def review(*, trial_dir: Path, instance_id: str, trial_result: dict) -> dict:
    """Qualify only the audited deSitter task; other ET and MSB tasks are unchanged."""
    result = {"version": "ainstein-desitter-sensitivity-v1", "metrics": {}, "evidence": {}}
    if instance_id != TASK:
        return result
    result["evidence"] = {
        "control_image": "sha256:c4884e5e26ca6ba2f0123d9869e4b36ca07a2bcbe7c56ee2b7b9c46fc698ab52",
        "reference_code_sha256": "1dc85633686763d8e810a4dacab6e7ffaf8d9dedfbbde63320842d972f87aa98",
        "perturbed_code_sha256": "7ad724524269e553b96792246e49c16b888c7c1bf4a9a9bd93751fa713aeefdf",
        "perturbation": "am=arad*t**(4.0D0/3.0D0) changed to am=2.0D0*arad*t**(4.0D0/3.0D0)",
        "mapped_tests": ["Schwarzschild_EF", "bowl-init"],
        "reference_creator_score": 100.0,
        "perturbed_creator_score": 100.0,
    }
    reason = (
        "The audited creator mapping gives 100 to both the reference and a doubled deSitter scale-factor formula. "
        "Its Schwarzschild_EF and bowl-init tests do not establish correctness of the requested deSitter formula. "
        "Preserve the raw creator metric; scientific correctness requires separate task-sensitive validation."
    )
    for name in (trial_result.get("verifier_result") or {}).get("rewards") or {}:
        result["metrics"][name] = {"status": "unresolved", "reason": reason}
    return result
