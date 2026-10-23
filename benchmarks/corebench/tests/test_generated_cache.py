# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import importlib.util
import marshal
import sys

import pytest

from benchmarks.corebench.generated_cache import audit


def fixture(tmp_path):
    source = "code/utils/example.py"
    cache = f"code/utils/__pycache__/example.{sys.implementation.cache_tag}.pyc"
    path = tmp_path / source
    path.parent.mkdir(parents=True)
    path.write_text("def answer():\n    return 42\n")
    p = tmp_path / cache
    p.parent.mkdir()
    p.write_bytes(b"original frozen cache")
    manifest = {s: hashlib.sha256((tmp_path / s).read_bytes()).hexdigest() for s in (source, cache)}
    filename = "/code/utils/example.py"
    p.write_bytes(
        importlib.util.MAGIC_NUMBER
        + bytes(12)
        + marshal.dumps(compile(path.read_bytes(), filename, "exec", dont_inherit=True, optimize=0))
    )
    return source, cache, manifest, {source: filename}


def test_valid_regenerated_cache_preserves_original_and_generated_hashes(tmp_path):
    source, cache, manifest, paths = fixture(tmp_path)
    result = audit(tmp_path, manifest, paths)
    assert result["changed"] == [cache]
    assert result["original_sha256"] == manifest
    assert result["observed_sha256"][source] == manifest[source]
    assert result["observed_sha256"][cache] != manifest[cache]
    assert result["caches"][0]["code_equivalent"] is True


@pytest.mark.parametrize(
    "mutation", ["source", "missing_source", "arbitrary_pyc", "header", "code", "filename", "missing_mapping"]
)
def test_reject_unproven_cache_changes(tmp_path, mutation):
    source, cache, manifest, paths = fixture(tmp_path)
    if mutation == "source":
        (tmp_path / source).write_text("changed = True\n")
    elif mutation == "missing_source":
        del manifest[source]
    elif mutation == "arbitrary_pyc":
        new = "code/other.pyc"
        (tmp_path / cache).rename(tmp_path / new)
        manifest[new] = manifest.pop(cache)
    elif mutation == "header":
        (tmp_path / cache).write_bytes(b"bad header")
    elif mutation in ("code", "filename"):
        text = "wrong = True" if mutation == "code" else (tmp_path / source).read_text()
        filename = paths[source] if mutation == "code" else "/different/path.py"
        (tmp_path / cache).write_bytes(
            importlib.util.MAGIC_NUMBER + bytes(12) + marshal.dumps(compile(text, filename, "exec"))
        )
    else:
        paths = {}
    with pytest.raises(ValueError):
        audit(tmp_path, manifest, paths)


def test_placement_cache_receipt_rejects_unbound_or_false_proof(tmp_path):
    import copy

    from benchmarks.corebench.placement_evidence import validate_generated_cache

    source, cache, manifest, paths = fixture(tmp_path)
    proof = audit(tmp_path, manifest, paths)
    after = {"files": len(manifest), "changed": [cache]}
    validate_generated_cache(manifest, after, proof)
    for mutation in ("false_equivalence", "source_hash", "missing_record", "generated_hash", "original_hash"):
        damaged = copy.deepcopy(proof)
        if mutation == "false_equivalence":
            damaged["caches"][0]["code_equivalent"] = False
        elif mutation == "source_hash":
            damaged["observed_sha256"][source] = "0" * 64
        elif mutation == "missing_record":
            damaged["caches"] = []
        elif mutation == "generated_hash":
            damaged["caches"][0]["generated_sha256"] = "0" * 64
        else:
            damaged["original_sha256"][cache] = "0" * 64
        with pytest.raises(ValueError):
            validate_generated_cache(manifest, after, damaged)


def test_full_placement_requires_bound_cache_proof_and_same_image(tmp_path):
    import json

    from benchmarks.corebench.placement_evidence import validate_gpu_placement
    from benchmarks.corebench.tests.test_frozen20 import gpu_evidence

    source, cache, manifest, paths = fixture(tmp_path)
    row = {"medium_file_manifest": [{"path": k, "sha256": v} for k, v in manifest.items()]}
    q = {"final_image": "sha256:final"}
    limits = {"cpus": 2, "memory_mb": 8192, "gpus": 1, "agent_timeout_sec": 3600}
    evidence = gpu_evidence(tmp_path, row, q)
    root = tmp_path / "gpu-control"
    (root / "after-source.json").write_text(json.dumps({"files": len(manifest), "changed": [cache]}))
    evidence["sha256"]["after-source.json"] = hashlib.sha256((root / "after-source.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="integrity"):
        validate_gpu_placement(row, q, limits, evidence)
    (root / "generated-cache.json").write_text(json.dumps(audit(tmp_path, manifest, paths)))
    evidence["sha256"]["generated-cache.json"] = hashlib.sha256(
        (root / "generated-cache.json").read_bytes()
    ).hexdigest()
    assert validate_gpu_placement(row, q, limits, evidence)["image"] == q["final_image"]
    with pytest.raises(ValueError, match="mismatch"):
        validate_gpu_placement(row, {"final_image": "sha256:different"}, limits, evidence)
