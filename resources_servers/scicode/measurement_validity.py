# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evidence-bound exclusion of the unchanged creator 79.4 shape conflict."""

import hashlib
import json
from pathlib import Path


def review_substep(sub_step: dict, *, data_sha256: str) -> dict | None:
    """Recognize only the reviewed prompt/assertions and canonical reference asset."""
    evidence = json.loads(Path(__file__).with_name("shape_review.json").read_text())
    digest = hashlib.sha256(json.dumps(sub_step, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if data_sha256 != evidence["h5_sha256"] or digest != evidence["step_sha256"]:
        return None
    return {
        "version": evidence["version"],
        "status": "unresolved",
        "reason": "Creator prompt requires column vectors but canonical targets are flat; unchanged comparison broadcasts NxN.",
        "step_sha256": digest,
        "h5_sha256": data_sha256,
        "target_shapes": evidence["target_shapes"],
        "raw_score": None,
        "executed": False,
        "historical_evidence": evidence["evidence"],
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
