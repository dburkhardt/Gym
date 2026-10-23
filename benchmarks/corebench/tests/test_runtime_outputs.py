# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json

import pytest

from benchmarks.corebench.placement_evidence import validate_gpu_placement
from benchmarks.corebench.runtime_outputs import POLICY
from benchmarks.corebench.tests.test_frozen20 import gpu_evidence


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "source",
        "data",
        "extra_path",
        "forged_source",
        "missing_proof",
        "missing_file",
        "image",
        "task",
        "observed_mismatch",
        "duplicate",
        "boolean",
        "cache",
        "missing_cache",
        "forged_cache_observation",
    ],
)
def test_full_placement_runtime_outputs_are_exactly_source_bound(tmp_path, mutation):
    original = dict(POLICY["source_sha256"])
    original.update({p: "1" * 64 for p in POLICY["outputs"]})
    original["data/original.csv"] = "2" * 64
    cache_proof = None
    if mutation in {"cache", "missing_cache", "forged_cache_observation"}:
        from benchmarks.corebench.generated_cache import audit
        from benchmarks.corebench.tests.test_generated_cache import fixture

        _, _, cache_manifest, paths = fixture(tmp_path)
        original.update(cache_manifest)
        cache_proof = audit(tmp_path, cache_manifest, paths)
    observed = dict(original)
    observed.update({p: "3" * 64 for p in POLICY["outputs"]})
    if cache_proof is not None:
        observed.update(cache_proof["observed_sha256"])
        remaining = {k: v for k, v in original.items() if k not in POLICY["outputs"]}
        cache_proof["original_sha256"] = remaining
        cache_proof["observed_sha256"] = {k: observed[k] for k in remaining}
    row = {
        "id": POLICY["capsule"],
        "published_image": POLICY["creator_image"],
        "instruction_sha256": POLICY["instruction_sha256"],
        "medium_file_manifest": [{"path": k, "sha256": v} for k, v in original.items()],
    }
    qualification = {"final_image": "sha256:final"}
    limits = {"cpus": 2, "memory_mb": 8192, "gpus": 1, "agent_timeout_sec": 3600}
    evidence = gpu_evidence(tmp_path, row, qualification)
    root = tmp_path / "gpu-control"
    receipt = {
        "schema": "core-creator-runtime-output/v1",
        "policy": copy.deepcopy(POLICY),
        "original_sha256": original,
        "observed_sha256": observed,
        "changed_output_paths": sorted(POLICY["outputs"]),
    }
    if mutation == "source":
        observed[next(iter(POLICY["source_sha256"]))] = "4" * 64
    elif mutation == "data":
        observed["data/original.csv"] = "4" * 64
    elif mutation == "extra_path":
        receipt["policy"]["outputs"]["data/original.csv"] = "arbitrary permission"
    elif mutation == "forged_source":
        receipt["policy"]["source_sha256"][next(iter(POLICY["source_sha256"]))] = "4" * 64
    elif mutation == "missing_file":
        del observed["data/original.csv"]
    elif mutation == "image":
        row["published_image"] = "sha256:other"
    elif mutation == "task":
        row["instruction_sha256"] = "4" * 64
    after = {
        "files": len(original),
        "changed": sorted(k for k in original if observed.get(k) != original[k]),
        "removed": [],
        "symlinks": {},
        "after_hashes": dict(observed),
    }
    if mutation == "observed_mismatch":
        after["after_hashes"][next(iter(POLICY["outputs"]))] = "5" * 64
    elif mutation == "duplicate":
        after["changed"].append(after["changed"][0])
    elif mutation == "boolean":
        after["files"] = True
    for name, value in [("after-source.json", after), ("runtime-outputs.json", receipt)]:
        if name == "runtime-outputs.json" and mutation == "missing_proof":
            continue
        path = root / name
        path.write_text(json.dumps(value))
        evidence["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if cache_proof is not None and mutation != "missing_cache":
        if mutation == "forged_cache_observation":
            cache_proof["observed_sha256"]["data/original.csv"] = "5" * 64
        path = root / "generated-cache.json"
        path.write_text(json.dumps(cache_proof))
        evidence["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if mutation in {None, "cache"}:
        assert validate_gpu_placement(row, qualification, limits, evidence)["image"] == qualification["final_image"]
        assert json.loads((root / "manifest.json").read_text()) == original
    else:
        with pytest.raises(ValueError):
            validate_gpu_placement(row, qualification, limits, evidence)
