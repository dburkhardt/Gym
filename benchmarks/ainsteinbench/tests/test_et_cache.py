# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib

import pytest

from benchmarks.ainsteinbench.et_cache import audit_object


def regular(data):
    return {"type": "0", "sha256": hashlib.sha256(data).hexdigest()}


def fixture(dependency="/src/other.cc /include/header.h"):
    obj = b"\x7fELF\x02\x01" + b"\0" * 10 + b"\x01\0" + b"\0" * 2 + b"immutable object"
    dep = f"other.cc.o other.cc.d: \\\n {dependency}\n".encode()
    original = {
        "build/other.cc.o": regular(obj),
        "build/other.cc.d": regular(dep),
        "src/other.cc": regular(b"other source"),
        "src/target.cc": regular(b"gold target"),
        "include/header.h": regular(b"header"),
    }
    return dict(
        object_path="/build/other.cc.o",
        object_bytes=obj,
        dependency_bytes=dep,
        original=original,
        candidate=copy.deepcopy(original),
        forbidden_sources=["/src/target.cc"],
    )


def test_exact_expanded_dependencies_are_bound():
    args = fixture()
    result = audit_object(**args)
    assert [x["canonical_path"] for x in result["dependencies"]] == ["src/other.cc", "include/header.h"]
    assert result["object_sha256"] == hashlib.sha256(args["object_bytes"]).hexdigest()
    assert result["scientific_qualification"] is False


@pytest.mark.parametrize("alias", ["/src/target.cc", "/alias/target.cc", "/duplicate.cc"])
def test_target_including_symlink_or_identical_source_rejected(alias):
    args = fixture(alias)
    for manifest in (args["original"], args["candidate"]):
        manifest["alias"] = {"type": "2", "linkname": "src"}
        manifest["duplicate.cc"] = regular(b"gold target")
    with pytest.raises(ValueError, match="Target-derived"):
        audit_object(**args)


@pytest.mark.parametrize("change", ["missing", "changed", "redirected"])
def test_candidate_dependency_must_remain_exact(change):
    args = fixture()
    if change == "missing":
        del args["candidate"]["include/header.h"]
    elif change == "changed":
        args["candidate"]["include/header.h"] = regular(b"different header")
    else:
        args["candidate"]["include/header.h"] = {"type": "2", "linkname": "/src/other.cc"}
    with pytest.raises(ValueError, match="dependency|Dependency"):
        audit_object(**args)


@pytest.mark.parametrize("path", ["/build/libthorn_GRHydro.a", "/exe/cactus_sim"])
def test_aggregate_archive_or_executable_is_not_an_object_receipt(path):
    args = fixture()
    args["object_path"] = path
    with pytest.raises(ValueError, match="archives are excluded"):
        audit_object(**args)


@pytest.mark.parametrize("field", ["object_bytes", "dependency_bytes"])
def test_mutated_actual_cache_bytes_fail(field):
    args = fixture()
    args[field] += b"tampered"
    with pytest.raises(ValueError, match="cache bytes changed"):
        audit_object(**args)


def test_missing_original_dependency_fails():
    args = fixture("/missing.cc")
    with pytest.raises(ValueError, match="Missing regular dependency"):
        audit_object(**args)


def test_symlink_parent_resolution_uses_real_filesystem_semantics():
    args = fixture("/alias/../target.cc")
    for manifest in (args["original"], args["candidate"]):
        manifest["alias"] = {"type": "2", "linkname": "/src/subdir"}
    with pytest.raises(ValueError, match="Target-derived"):
        audit_object(**args)


@pytest.mark.parametrize(
    "dep", [b"other.o: /src/other.cc", b"other.cc.o other.cc.d: $(SOURCE)", b"other.cc.o other.cc.d:"]
)
def test_unbound_or_ambiguous_dependency_rules_fail(dep):
    args = fixture()
    args["dependency_bytes"] = dep
    args["original"]["build/other.cc.d"] = regular(dep)
    with pytest.raises(ValueError):
        audit_object(**args)


def test_fortran_empty_same_object_rule_preserves_all_dependencies():
    args = fixture()
    args["dependency_bytes"] += b"other.cc.o:\n"
    args["original"]["build/other.cc.d"] = regular(args["dependency_bytes"])
    assert len(audit_object(**args)["dependencies"]) == 2


@pytest.mark.parametrize("extra", [b"other.cc.o: /src/target.cc", b"unrelated.o:"])
def test_extra_dependency_rule_is_never_discarded(extra):
    args = fixture()
    args["dependency_bytes"] += extra
    args["original"]["build/other.cc.d"] = regular(args["dependency_bytes"])
    with pytest.raises(ValueError, match="syntax"):
        audit_object(**args)


@pytest.mark.parametrize("data", [b"!<arch>\n" + b"x" * 30, b"\x7fELF\x02\x01" + b"\0" * 10 + b"\x02\0" + b"x" * 20])
def test_archive_or_executable_disguised_as_object_rejected(data):
    args = fixture()
    args["object_bytes"] = data
    args["original"]["build/other.cc.o"] = regular(data)
    with pytest.raises(ValueError, match="ELF relocatable"):
        audit_object(**args)
