# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run pinned upstream GenoTEX metrics outside the solver sandbox, retaining missingness."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def grade(upstream: Path, staging: Path, prediction: Path, reference: Path, trait: str, condition: str | None) -> dict:
    """Keep native stage metrics; reward is analysis F1/100, never an invented composite."""
    for path in prediction.rglob("*"):
        if path.is_symlink():
            raise ValueError("Solver output contains a symlink; refusing host-side grading")
    sys.path.insert(0, str(upstream))
    spec = importlib.util.spec_from_file_location("genotex_upstream_eval", upstream / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The upstream script otherwise enumerates every metadata pair and always adds
    # unconditional pairs. This changes only the selection of problems, not any metric.
    module.get_question_pairs = lambda _path: [(trait, condition)]
    previous = Path.cwd()
    try:
        os.chdir(staging)
        raw = module.main(
            str(prediction), str(reference), ["selection", "preprocessing", "analysis"], ["gene", "clinical", "linked"]
        )
    finally:
        os.chdir(previous)
    reference_files = [p.relative_to(reference) for p in reference.rglob("*") if p.is_file()]
    # Reference bundles also ship implementation scripts. Only the documented
    # JSON/CSV outputs participate in artifact completeness.
    expected = [p for p in reference_files if p.suffix in (".json", ".csv") and "code" not in p.parts]
    auxiliary = [str(p) for p in reference_files if p not in expected]
    missing = [str(p) for p in expected if not (prediction / p).is_file()]
    analysis = raw.get("analysis", {}).get("categorized", {}).get("Overall", {})
    f1 = analysis.get("f1")
    reward = float(f1) / 100 if isinstance(f1, (float, int)) and math.isfinite(f1) else 0.0
    return {
        "reward": reward,
        "reward_metric": "upstream analysis F1 / 100; missing/undefined => 0 for Gym only",
        "raw_upstream": json_safe(raw),
        "expected_files": len(expected),
        "auxiliary_reference_files": auxiliary,
        "missing_files": missing,
        "complete": not missing,
        "requested_pairs": 1,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["upstream", "staging", "prediction", "reference", "result"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--condition", default="None")
    parser.add_argument("--identity", type=Path)
    args = parser.parse_args()
    if args.identity:
        paths = [Path(__file__).resolve(), args.upstream.resolve() / "eval.py", Path(sys.executable).resolve()]
        args.identity.write_text(
            json.dumps(
                {
                    "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    "python_version": sys.version,
                },
                indent=2,
            )
        )
    result = grade(
        args.upstream.resolve(),
        args.staging.resolve(),
        args.prediction.resolve(),
        args.reference.resolve(),
        args.trait,
        None if args.condition == "None" else args.condition,
    )
    args.result.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
