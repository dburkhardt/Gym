# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute exact pinned creator prompt/parser/scorer without importing its API client."""

import ast
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


PIN = "09d7a6bb68e7dc04167ad57c653c4ea9c8f1702e"


def creator(root):
    root = Path(root)
    if subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() != PIN:
        raise ValueError("CTBench source pin mismatch")
    source = (root / "module.py").read_bytes()
    tree = ast.parse(source)
    names = {
        "build_zeroshot_prompt",
        "get_question_from_row",
        "build_gpt4_eval_prompt",
        "extract_elements_v2",
        "match_to_score",
    }
    module = ast.Module(
        body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]
    )
    namespace = {"re": re}
    exec(compile(module, str(root / "module.py"), "exec"), namespace)
    namespace["__creator_source_sha256"] = hashlib.sha256(source).hexdigest()
    return namespace


def parse_and_score(text, references, candidates, functions):
    result = json.loads(text)
    pairs = result["matched_features"]
    refs = result["remaining_reference_features"]
    cands = result["remaining_candidate_features"]
    if not isinstance(pairs, list) or any(not isinstance(p, list) or len(p) != 2 for p in pairs):
        raise ValueError("Malformed matching pairs")
    # Integrity guard is additional to creator scoring: refuse hallucinated or omitted features.
    if Counter([p[0] for p in pairs] + refs) != Counter(references):
        raise ValueError("Judge reference coverage mismatch")
    if Counter([p[1] for p in pairs] + cands) != Counter(candidates):
        raise ValueError("Judge candidate coverage mismatch")
    # Preserve creator zero-denominator exception rather than silently repairing it.
    return functions["match_to_score"](pairs, refs, cands)
