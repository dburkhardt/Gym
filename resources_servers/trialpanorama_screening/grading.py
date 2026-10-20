# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned creator study screening, including raw malformed-output behavior."""

import ast
import copy
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def creator(root):
    root = Path(root)
    pin = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if pin != "20e97cbb823003fccc425732d593cf2bb205d163":
        raise ValueError("Source pin mismatch")
    tree = ast.parse((root / "benchmark/tasks/study_screening.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StudyScreeningTask")
    keep = {"_default_system_prompt", "prepare_input", "parse_prediction", "evaluate_prediction"}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in keep]
    cls.bases = []
    ns = {"json": json, "copy": copy, "Any": Any, "Dict": Dict, "List": List, "Optional": Optional}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "pinned_trialpanorama", "exec"), ns)
    return ns["StudyScreeningTask"]()


def grade(text: str, *, creator_root: str, sample: dict, **metadata: object) -> dict:
    task = creator(creator_root)
    result = task.evaluate_prediction(sample, text)
    try:
        value = json.loads(text)
        parsed = value.get("results", []) if isinstance(value, dict) else value
        valid = (
            isinstance(parsed, list)
            and bool(parsed)
            and all(isinstance(x, dict) and x.get("decision") in {"included", "excluded"} for x in parsed)
        )
        supplied = [x.get("PMID") for x in parsed] if valid else []
        complete = set(supplied) == {x["PMID"] for x in sample["labels"]} and len(supplied) == len(set(supplied))
    except (ValueError, TypeError):
        valid = False
        complete = False
    return {
        "reward": result["micro_accuracy"],
        "diagnostics": {
            "creator": result,
            "valid_json_decisions": valid,
            "complete_pmid_coverage": complete,
            "strict_usable": valid and complete,
            "protocol": "creator study_screening raw micro accuracy; report per-study aggregate separately",
        },
    }
