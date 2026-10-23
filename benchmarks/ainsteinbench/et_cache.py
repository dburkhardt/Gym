# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed audit of individual compiler caches; never restores or builds them.

Manifests must come from independently verified immutable image readbacks. A receipt
binds compiler dependency records and object bytes, not arbitrary opaque libraries.
It does not prove compiler correctness or authorize scientific runtime qualification.
"""

import hashlib
import posixpath
import shlex
from collections.abc import Mapping, Sequence
from typing import Any


Manifest = Mapping[str, Mapping[str, Any]]


def _canonical(path: str, manifest: Manifest) -> str:
    if not path.startswith("/"):
        raise ValueError("Compiler dependency must be absolute")
    remaining = path.split("/")[1:]
    resolved: list[str] = []
    links = 0
    while remaining:
        part = remaining.pop(0)
        if part in ("", "."):
            continue
        if part == "..":
            if not resolved:
                raise ValueError("Dependency escapes filesystem root")
            resolved.pop()
            continue
        resolved.append(part)
        entry = manifest.get("/".join(resolved), {})
        if entry.get("type") in ("1", "2"):
            links += 1
            if links > 40:
                raise ValueError("Dependency link cycle")
            target = entry.get("linkname")
            if not isinstance(target, str) or not target:
                raise ValueError("Missing dependency link target")
            resolved.pop()
            if entry["type"] == "1" or target.startswith("/"):
                resolved = []
            remaining = target.split("/") + remaining
    result = "/".join(resolved)
    if manifest.get(result, {}).get("type") != "0":
        raise ValueError(f"Missing regular dependency: {result}")
    return result


def _digest(entry: Mapping[str, Any]) -> str:
    value = entry.get("sha256")
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Missing valid manifest SHA256")
    return value


def audit_object(
    *,
    object_path: str,
    object_bytes: bytes,
    dependency_bytes: bytes,
    original: Manifest,
    candidate: Manifest,
    forbidden_sources: Sequence[str],
) -> dict[str, Any]:
    """Bind one .o and its .d to unchanged, non-target expanded dependencies.

    Aggregated archives/executables are deliberately unsupported. Callers must not
    infer their safety from member filenames or use a receipt as a blanket cache
    allowlist. Symlink resolution follows the filesystem, including symlink/..
    semantics. Unsupported make syntax is rejected rather than guessed.
    """
    if not object_path.startswith("/") or not object_path.endswith(".o"):
        raise ValueError("Only individual .o files are supported; archives are excluded")
    if not forbidden_sources:
        raise ValueError("Explicit forbidden source binding is required")
    object_key = _canonical(object_path, original)
    if not object_key.endswith(".o"):
        raise ValueError("Object resolves to a non-object artifact")
    if (
        len(object_bytes) < 20
        or object_bytes[:4] != b"\x7fELF"
        or object_bytes[4] not in (1, 2)
        or object_bytes[5] not in (1, 2)
        or int.from_bytes(object_bytes[16:18], "little" if object_bytes[5] == 1 else "big") != 1
    ):
        raise ValueError("Cache must be an ELF relocatable object, not an archive/executable")
    dependency_path = object_path[:-1] + "d"
    dependency_key = _canonical(dependency_path, original)
    for key, data in ((object_key, object_bytes), (dependency_key, dependency_bytes)):
        if hashlib.sha256(data).hexdigest() != _digest(original[key]):
            raise ValueError(f"Original cache bytes changed: {key}")
    forbidden = {_canonical(path, original) for path in forbidden_sources}
    forbidden_hashes = {_digest(original[path]) for path in forbidden}
    text = dependency_bytes.decode("utf-8").replace("\\\n", " ").strip()
    # The pinned Fortran preprocessor appends an empty rule for the same .o.
    # Only this exact no-op is supported; never ignore another rule's dependencies.
    lines = text.splitlines()
    if len(lines) == 2 and lines[1].strip() == posixpath.basename(object_path) + ":":
        text = lines[0]
    if any(token in text for token in ("$", "#", "\n", "\r")) or text.count(":") != 1:
        raise ValueError("Unsupported compiler dependency syntax")
    targets_text, dependencies_text = text.split(":", 1)
    targets = shlex.split(targets_text)
    if len(targets) != 2 or set(targets) != {posixpath.basename(object_path), posixpath.basename(dependency_path)}:
        raise ValueError("Dependency targets do not bind this object")
    dependencies = shlex.split(dependencies_text)
    if not dependencies:
        raise ValueError("Empty compiler dependencies")
    bindings = []
    for dependency in dependencies:
        old = _canonical(dependency, original)
        old_hash = _digest(original[old])
        if old in forbidden or old_hash in forbidden_hashes:
            raise ValueError(f"Target-derived cache dependency: {dependency}")
        new = _canonical(dependency, candidate)
        if new != old or _digest(candidate[new]) != old_hash:
            raise ValueError(f"Dependency changed in candidate image: {dependency}")
        bindings.append({"path": dependency, "canonical_path": old, "sha256": old_hash})
    return {
        "schema": "ainstein-et-object-dependencies/v1",
        "object_path": object_key,
        "object_sha256": hashlib.sha256(object_bytes).hexdigest(),
        "dependency_path": dependency_key,
        "dependency_sha256": hashlib.sha256(dependency_bytes).hexdigest(),
        "dependencies": bindings,
        "forbidden_sources": sorted(forbidden),
        "aggregate_archives_supported": False,
        "scientific_qualification": False,
    }
