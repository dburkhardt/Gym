# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind a separate owned GPU workflow control without rewriting CPU image qualification."""

import hashlib
import json
from pathlib import Path


REQUIRED = {
    "admission.json",
    "container.json",
    "manifest.json",
    "before-source.json",
    "after-source.json",
    "execution.json",
    "independent-cleanup.json",
    "stdout.txt",
}


def validate_gpu_placement(row, qualification, limits, evidence):
    """Validate immutable control receipts; this attests placement, not model performance."""
    if not isinstance(evidence, dict) or set(evidence) != {"root", "sha256"}:
        raise ValueError("GPU execution requires separate placement evidence")
    root = Path(evidence["root"])
    hashes = evidence["sha256"]
    if not isinstance(hashes, dict) or set(hashes) != REQUIRED:
        raise ValueError("Incomplete GPU placement evidence")
    for name, expected in hashes.items():
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("GPU placement evidence changed")
    read = lambda name: json.loads((root / name).read_text())
    admission, container = read("admission.json"), read("container.json")
    expected_sources = {f["path"]: f["sha256"] for f in row["medium_file_manifest"]}
    if read("manifest.json") != expected_sources:
        raise ValueError("GPU placement source differs")
    for name in ("before-source.json", "after-source.json"):
        if read(name) != {"files": len(expected_sources), "changed": []}:
            raise ValueError("GPU placement source integrity failed")
    expected = {
        "image": qualification["final_image"],
        "cpus": limits["cpus"],
        "memory_mib": limits["memory_mb"],
        "gpu": limits["gpus"],
        "scientific_timeout_s": limits["agent_timeout_sec"],
        "no_model_calls": True,
    }
    if any(type(admission.get(k)) is not type(v) or admission[k] != v for k, v in expected.items()):
        raise ValueError("GPU placement resource/image/protocol mismatch")
    host = container["HostConfig"]
    requests = host.get("DeviceRequests") or []
    if (
        container["Image"] != qualification["final_image"]
        or host["NanoCpus"] != limits["cpus"] * 10**9
        or host["Memory"] != limits["memory_mb"] * 1024**2
        or host["NetworkMode"] != "none"
        or len(requests) != 1
        or requests[0].get("Count") != limits["gpus"]
        or requests[0].get("DeviceIDs")
        or requests[0].get("Capabilities") != [["gpu"]]
    ):
        raise ValueError("Actual GPU container placement differs")
    result, cleanup = read("execution.json"), read("independent-cleanup.json")
    if (
        result.get("return_code") != 0
        or result.get("error_type") is not None
        or cleanup.get("absent") is not True
        or cleanup.get("sandbox_id") != container["Name"].lstrip("/")
        or not (root / "stdout.txt").stat().st_size
    ):
        raise ValueError("GPU workflow or exact owned cleanup incomplete")
    return {"sha256": dict(hashes), "image": qualification["final_image"], "resources": dict(limits)}
