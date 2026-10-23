# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load checksum-pinned creator methods without Docker client import side effects."""

import ast
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace


def checked_source(root: Path):
    hashes = json.loads(Path(__file__).with_name("source-sha256.json").read_text())
    for relative, expected in hashes.items():
        path = root / relative
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"AInstein creator source mismatch: {relative}")
    rows = []
    for name in ("msb", "et"):
        rows.extend(json.loads(x) for x in (root / f"data/questions/{name}_converted.jsonl").read_text().splitlines())
    if len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate creator task IDs")
    return {r["question_id"]: r for r in rows}, hashes


def creator_methods(root: Path):
    checked_source(root)
    tree = ast.parse((root / "scripts/run_agent.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DockerAgent")
    names = {"_build_msb_prompt", "_build_et_prompt", "_extract_code_from_markdown"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    tools = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TOOLS" for t in n.targets)
    )
    # Only annotations are deferred; creator function bodies remain unchanged.
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), tools, *methods],
        type_ignores=[],
    )
    namespace = {"logging": logging}
    exec(compile(ast.fix_missing_locations(module), str(root / "scripts/run_agent.py"), "exec"), namespace)
    return namespace


def prompt(root: Path, raw: dict) -> str:
    methods = creator_methods(root)
    if raw["question_type"] == "multi_swe_bench":
        question = SimpleNamespace(
            **{**raw, "environment": {**raw["environment"], "working_directory": "/home/pyscf"}}
        )
        return methods["_build_msb_prompt"](None, question)
    if raw["question_type"] == "einstein_toolkit":
        return methods["_build_et_prompt"](None, SimpleNamespace(**raw))
    raise ValueError("Unsupported AInstein question type")


def parse_log(root: Path, text: str):
    checked_source(root)
    path = root / "curation/msb/pyscf/pyscf.py"
    tree = ast.parse(path.read_text())
    functions = [
        n
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "parse_log"
    ]
    if len(functions) != 1:
        raise ValueError("Pinned PySCF parser identity ambiguous")
    # Creator parser constructs six counts/sets. Preserve these exactly; the
    # dependency's dataclass serialization layer is unnecessary here.
    namespace = {"TestResult": SimpleNamespace}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(path), "exec"), namespace)
    result = namespace["parse_log"](None, text)
    return {k: sorted(v) if isinstance(v, set) else v for k, v in vars(result).items()}
