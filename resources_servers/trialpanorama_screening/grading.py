# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned creator study screening, including raw malformed-output behavior."""

import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def creator(root):
    root = Path(root)
    pin = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if pin != "20e97cbb823003fccc425732d593cf2bb205d163":
        raise ValueError("Source pin mismatch")
    source = root / "benchmark/tasks/study_screening.py"
    committed = subprocess.check_output(["git", "-C", str(root), "show", "HEAD:benchmark/tasks/study_screening.py"])
    if source.read_bytes() != committed:
        raise ValueError("Creator scorer working tree differs from pinned source")
    tree = ast.parse(source.read_text())
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
            and all(
                isinstance(x, dict)
                and isinstance(x.get("decision"), str)
                and x["decision"].lower() in {"included", "excluded"}
                for x in parsed
            )
        )
        supplied = [x.get("PMID") for x in parsed] if valid else []
        complete = set(supplied) == {x["PMID"] for x in sample["labels"]} and len(supplied) == len(set(supplied))
        # Match creator confidence conversion: malformed confidence triggers fallback.
        if valid:
            for item in parsed:
                float(item.get("confidence", 50))
    except (ValueError, TypeError):
        valid = False
        complete = False
    usable = valid and complete
    reference_ids = [item["PMID"] for item in sample["labels"]]
    reference_duplicate = len(reference_ids) != len(set(reference_ids))
    digest = lambda value: hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source = Path(creator_root) / "benchmark/tasks/study_screening.py"
    review = {
        "version": "trialpanorama-complete-output-v1",
        "implementation": {
            "import_path": "resources_servers.trialpanorama_screening.grading:grade",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "evidence": {
            "creator_commit": "20e97cbb823003fccc425732d593cf2bb205d163",
            "creator_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "sample_sha256": digest(sample),
            "answer_sha256": digest(text),
            "strict_usable": usable,
        },
        "metrics": {
            "creator_micro_accuracy": {
                "status": "valid" if usable else "invalid",
                "reason": "Complete explicit decisions; creator comparison executed"
                if usable
                else "Creator fallback or incomplete/duplicate study coverage is not complete prediction evidence",
            },
            "complete_output_micro_accuracy_v1": {
                "status": "valid",
                "reason": "Explicit complete-output variant: unusable predictions are model failures scored zero; otherwise retain creator accuracy",
            },
        },
    }
    if reference_duplicate:
        for metric in review["metrics"].values():
            metric.update(
                status="unresolved",
                reason="Duplicate reference PMID: input multiplicity/label policy requires review; not attributable to model failure",
            )
    review["evidence"]["duplicate_reference_pmid"] = reference_duplicate
    return {
        "reward": result["micro_accuracy"],
        "metadata": {
            "source": "trialpanorama_screening",
            "measurement_review": review,
            "verifier_result": {
                "rewards": {
                    "creator_micro_accuracy": result["micro_accuracy"],
                    "complete_output_micro_accuracy_v1": result["micro_accuracy"] if usable else 0.0,
                }
            },
        },
        "mask_sample": not usable or reference_duplicate,
        "failure_kind": "trialpanorama:reference_duplicate"
        if reference_duplicate
        else (None if usable else "trialpanorama:model_invalid_screening_output"),
        "failure_reason": "Duplicate reference PMID; review required"
        if reference_duplicate
        else None
        if usable
        else "Missing, malformed, duplicate or incomplete explicit study decisions; raw creator fallback retained",
        "diagnostics": {
            "creator": result,
            "valid_json_decisions": valid,
            "complete_pmid_coverage": complete,
            "strict_usable": valid and complete,
            "protocol": "creator study_screening raw micro accuracy; report per-study aggregate separately",
        },
    }
