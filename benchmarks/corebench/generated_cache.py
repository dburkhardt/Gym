# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit regenerated CPython caches inside the original isolated interpreter.

This module is deliberately standard-library-only and Python 3.7 compatible. Never
unmarshal benchmark bytecode on the controller: run this probe in the owned,
network-disabled scientific container, with its original interpreter.
"""

import hashlib
import importlib.util
import json
import marshal
import re
import sys
import types
from pathlib import Path, PurePosixPath


def cache_source(path: str, tag: str) -> tuple:
    """Map only canonical CPython __pycache__ paths to their source and optimization."""
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or p.parent.name != "__pycache__":
        raise ValueError("Unsupported generated-cache path")
    match = re.fullmatch(r"(.+)\." + re.escape(tag) + r"(?:\.opt-([12]))?\.pyc", p.name)
    if not match:
        raise ValueError("Cache interpreter tag differs or path is unsupported")
    return str(p.parent.parent / (match[1] + ".py")), int(match[2] or 0)


def _same_code(actual: types.CodeType, expected: types.CodeType) -> bool:
    if not isinstance(actual, types.CodeType) or actual != expected or actual.co_filename != expected.co_filename:
        return False
    return all(
        _same_code(a, b)
        for a, b in zip(actual.co_consts, expected.co_consts)
        if isinstance(a, types.CodeType) or isinstance(b, types.CodeType)
    )


def audit(root: Path, manifest: dict, compile_paths: dict) -> dict:
    """Bind all originals and prove only regenerated caches differ from frozen bytes.

    compile_paths maps each changed cache's source to the original absolute Python
    compilation filename. It must come from the reviewed workflow layout, not from
    accepting an arbitrary filename embedded in the cache.
    """
    observed = {}
    for name, digest in manifest.items():
        p = PurePosixPath(name)
        path = root / name
        if p.is_absolute() or ".." in p.parts or path.is_symlink() or not path.is_file():
            raise ValueError("Missing, linked, or unsafe frozen input")
        if any(parent.is_symlink() for parent in path.parents if parent != root.parent):
            raise ValueError("Linked input ancestry")
        observed[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    changed = sorted(name for name in manifest if observed[name] != manifest[name])
    records = []
    sources = set()
    for name in changed:
        source, optimization = cache_source(name, sys.implementation.cache_tag)
        if source not in manifest or observed[source] != manifest[source]:
            raise ValueError("Generated cache has no unchanged frozen source")
        filename = compile_paths.get(source)
        if not isinstance(filename, str) or not PurePosixPath(filename).is_absolute():
            raise ValueError("Original compile filename must be explicit")
        sources.add(source)
        data = (root / name).read_bytes()
        if (
            len(data) < 16
            or data[:4] != importlib.util.MAGIC_NUMBER
            or int.from_bytes(data[4:8], "little") not in (0, 1, 3)
        ):
            raise ValueError("Unverifiable cache header/interpreter")
        code = marshal.loads(data[16:])
        expected = compile((root / source).read_bytes(), filename, "exec", dont_inherit=True, optimize=optimization)
        if not _same_code(code, expected):
            raise ValueError("Generated cache code differs from unchanged source")
        records.append(
            {
                "path": name,
                "source": source,
                "original_sha256": manifest[name],
                "generated_sha256": observed[name],
                "source_sha256": manifest[source],
                "compile_filename": filename,
                "optimization": optimization,
                "code_equivalent": True,
            }
        )
    if set(compile_paths) != sources:
        raise ValueError("Compile path mapping must match changed cache sources exactly")
    return {
        "schema": "core-generated-cache/v1",
        "python": sys.version,
        "cache_tag": sys.implementation.cache_tag,
        "magic_hex": importlib.util.MAGIC_NUMBER.hex(),
        "original_sha256": manifest,
        "observed_sha256": observed,
        "changed": changed,
        "caches": records,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--compile-paths", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                Path(args.root),
                json.loads(Path(args.manifest).read_text()),
                json.loads(Path(args.compile_paths).read_text()),
            ),
            indent=2,
        )
    )
