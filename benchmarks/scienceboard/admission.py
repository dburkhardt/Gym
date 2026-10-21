# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline validation of existing image evidence; never loads or executes an image."""

import argparse
import hashlib
import json
import re
from pathlib import Path


SOURCE_REVISION = "21b5eb0f1d00898f7de9e84c3d33a87c2f2b3c3e"
AUDIT_SHA256 = "40d420cb4f749092080138ea8926bcdbdb7a65cc20f21152b1b5a3b9aa923d80"
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def admit(metadata, audit, resources):
    """Validate operator-supplied docker inspect and creator audit receipts.

    Receipts attest an earlier audit; this does not authenticate their producer or
    replace actual runtime qualification. Capacity values are placement promises,
    not live host observations.
    """
    errors = []
    image_id = metadata.get("Id", "")
    if not DIGEST.fullmatch(image_id):
        errors.append("missing immutable Docker image Id")
    if metadata.get("Architecture") != "amd64" or metadata.get("Os") != "linux":
        errors.append("requires linux/amd64 image")
    config = metadata.get("Config") or {}
    if (config.get("Labels") or {}).get("org.scienceboard.distribution") != "sanitized-flat":
        errors.append("requires creator sanitized-flat label")
    if len((metadata.get("RootFS") or {}).get("Layers", [])) != 1:
        errors.append("requires exactly one flattened filesystem layer")
    for item in config.get("Env") or []:
        name, _, value = item.partition("=")
        if value and re.search(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", name, re.I):
            errors.append("image contains credential-like environment setting")
    if audit.get("source_revision") != SOURCE_REVISION or audit.get("script_sha256") != AUDIT_SHA256:
        errors.append("creator audit source revision mismatch")
    report = audit.get("report") or {}
    if audit.get("returncode") != 0 or audit.get("image_id") != image_id or report.get("image") != image_id:
        errors.append("audit must succeed against this immutable image Id, not a mutable tag")
    if report.get("passed") is not True or report.get("findings") != [] or report.get("finding_count") != 0:
        errors.append("missing or failed creator sanitization audit")
    for key, minimum in {"cpus": 8, "memory_gib": 24, "free_disk_gib": 80, "shm_gib": 2}.items():
        value = resources.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= value < float("inf"):
            errors.append(f"placement requires {key} >= {minimum}")
    binding = {"metadata": metadata, "audit": audit, "resources": resources}
    return {
        "status": "image_evidence_admitted_runtime_unqualified" if not errors else "blocked",
        "runnable": False,
        "image_id": image_id,
        "source_revision": SOURCE_REVISION,
        "evidence_sha256": hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest(),
        "blockers": errors,
        "remaining_gates": [
            "actual application/task initializer controls",
            "frozen task binding",
            "native guarded agent and verifier qualification",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    args = parser.parse_args()
    try:
        metadata = json.loads(args.inspect.read_text())
        if isinstance(metadata, list):
            if len(metadata) != 1:
                raise ValueError("expected exactly one Docker image")
            metadata = metadata[0]
        result = admit(metadata, json.loads(args.audit.read_text()), json.loads(args.resources.read_text()))
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        result = {"status": "blocked", "runnable": False, "blockers": [str(exc)]}
    print(json.dumps(result, indent=2))
    return 0 if result["status"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
