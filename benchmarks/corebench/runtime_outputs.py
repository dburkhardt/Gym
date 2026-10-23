# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit creator-authored output mutations, never an arbitrary source exclusion."""

import re


POLICY = {
    "capsule": "capsule-4645832",
    "creator_image": "registry.codeocean.com/published/994080f7-78c8-4edf-9363-24009fcaba50@sha256:9b86f2f634b2a9e102c7d205bb3a322770a4a181ee7f7547e731d4dbf32debbc",
    "instruction_sha256": "f9db1f3b72d5e44f65a440a63d2478be92fd5ad4bc6af8608041d6f6f4d558bf",
    "source_sha256": {
        "code/models/CausalModel.py": "6c77790a315e3006b9e81d4b30e8d41feb09828923c3f55fc25ee8e0c8033805",
        "code/models/GNN_TARnet_hyper.py": "a1ab606eb9f08283b035bc624c16291355cd859112de3c04ded71db700d7987c",
    },
    "outputs": {
        "code/Emissions/GNNTARnet/default/GNNTARnet_ihdp_a_train.csv": "Creator CausalModel.py817-850 deletes/recreates this default train emissions output for count0.",
        "code/Emissions/GNNTARnet/default/GNNTARnet_ihdp_a_test.csv": "Creator CausalModel.py817-850 deletes/recreates this default test emissions output for count0.",
    },
}


def validate_runtime_outputs(row, original, after, receipt):
    """Return the complete non-output manifest only after exact policy/hash validation.

    Callers must still validate every remaining original, including generated
    cache equivalence. The full original and observed maps remain in the receipt.
    """
    if (
        receipt.get("schema") != "core-creator-runtime-output/v1"
        or receipt.get("policy") != POLICY
        or row["id"] != POLICY["capsule"]
        or row["published_image"] != POLICY["creator_image"]
        or row["instruction_sha256"] != POLICY["instruction_sha256"]
        or receipt.get("original_sha256") != original
        or type(after.get("files")) is not int
        or after["files"] != len(original)
    ):
        raise ValueError("Runtime output policy/source/task binding differs")
    observed = receipt.get("observed_sha256")
    if not isinstance(observed, dict) or set(observed) != set(original):
        raise ValueError("Runtime output receipt omitted original files")
    if any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in observed.values()):
        raise ValueError("Invalid observed runtime hash")
    actual = after.get("after_hashes")
    if not isinstance(actual, dict) or {k: actual.get(k) for k in original} != observed:
        raise ValueError("Runtime output observation differs from actual after-source hashes")
    for name, expected in POLICY["source_sha256"].items():
        if original.get(name) != expected or observed.get(name) != expected:
            raise ValueError("Runtime output creator source changed")
    outputs = set(POLICY["outputs"])
    if not outputs <= set(original):
        raise ValueError("Original runtime output files missing")
    changed = sorted(name for name in original if observed[name] != original[name])
    if (
        not isinstance(after.get("changed"), list)
        or any(not isinstance(p, str) for p in after["changed"])
        or len(set(after["changed"])) != len(after["changed"])
        or sorted(after["changed"]) != changed
    ):
        raise ValueError("Runtime output change census differs")
    if receipt.get("changed_output_paths") != sorted(outputs & set(changed)):
        raise ValueError("Runtime output changed path set differs")
    if after.get("symlinks") or after.get("removed"):
        raise ValueError("Runtime output files removed or replaced with links")
    remaining = {k: v for k, v in original.items() if k not in outputs}
    residual = [name for name in changed if name not in outputs]
    return remaining, {"files": len(remaining), "changed": residual}, {k: observed[k] for k in remaining}
