# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind a separate owned GPU workflow control without rewriting CPU image qualification."""

import hashlib
import json
from pathlib import Path

from benchmarks.corebench.generated_cache import cache_source


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
    if not isinstance(hashes, dict) or set(hashes) not in (REQUIRED, REQUIRED | {"generated-cache.json"}):
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
    if read("before-source.json") != {"files": len(expected_sources), "changed": []}:
        raise ValueError("GPU placement source integrity failed")
    after = read("after-source.json")
    if after != {"files": len(expected_sources), "changed": []}:
        if "generated-cache.json" not in hashes:
            raise ValueError("GPU placement source integrity failed")
        validate_generated_cache(expected_sources, after, read("generated-cache.json"))
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


def validate_generated_cache(original: dict, after: dict, proof: dict) -> None:
    """Validate hash-bound isolated-interpreter evidence without loading bytecode here."""
    changed = after.get("changed")
    observed = proof.get("observed_sha256")
    if (
        after.get("files") != len(original)
        or not isinstance(changed, list)
        or not changed
        or len(set(changed)) != len(changed)
        or proof.get("schema") != "core-generated-cache/v1"
        or proof.get("original_sha256") != original
        or proof.get("changed") != sorted(changed)
        or not isinstance(observed, dict)
        or set(observed) != set(original)
        or sorted(k for k in original if observed[k] != original[k]) != sorted(changed)
        or not isinstance(proof.get("cache_tag"), str)
        or not proof.get("python")
        or not proof.get("magic_hex")
    ):
        raise ValueError("Generated-cache proof does not bind original integrity evidence")
    records = proof.get("caches")
    if not isinstance(records, list) or len(records) != len(changed):
        raise ValueError("Incomplete generated-cache proof")
    seen = set()
    for record in records:
        name = record.get("path")
        if name not in changed or name in seen:
            raise ValueError("Duplicate or unrelated cache proof")
        seen.add(name)
        source, optimization = cache_source(name, proof["cache_tag"])
        if (
            source not in original
            or observed[source] != original[source]
            or record.get("source") != source
            or record.get("original_sha256") != original[name]
            or record.get("generated_sha256") != observed[name]
            or record.get("source_sha256") != original[source]
            or record.get("optimization") != optimization
            or record.get("code_equivalent") is not True
            or not isinstance(record.get("compile_filename"), str)
            or not record["compile_filename"].startswith("/")
        ):
            raise ValueError("Unproven generated-cache equivalence")
