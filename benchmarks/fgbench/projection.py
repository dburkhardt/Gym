# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Project creator metrics without pooling accuracy and numeric transport rewards."""

import hashlib
import json
from pathlib import Path

from benchmarks.scientific_agent_campaign import deterministic


SCORER_SHA256 = "5c808b25ad08b376dbb29e8d3739e52309b787a0be0fabcc2ecced8a7262ffe9"


def project(repo: Path, task: dict, details: dict, *, protocol: dict) -> dict:
    scorer = repo / "result_ananlysis/analysis.py"
    actual = hashlib.sha256(scorer.read_bytes()).hexdigest()
    if actual != SCORER_SHA256:
        raise ValueError("FGBench metric projection requires the inspected pinned creator scorer bytes")
    upstream = deterministic.module(scorer)
    stratum = upstream.task_name(task["verifier_metadata"])
    boolean = stratum.endswith("bool")
    valid = details["valid"] if boolean else details["creator_valid"]
    value = details["reward"] if boolean else details["creator_squared_error"]
    metric = f"fgbench/{stratum}/" + ("accuracy" if boolean else "rmse")
    validity = f"fgbench/{stratum}/valid_rate"
    reason = (
        "Creator boolean accuracy includes incorrect and missing model answers"
        if boolean
        else "Creator RMSE includes only nonzero numeric truth/prediction pairs"
    )
    return {
        "source": "fgbench",
        "config": {"metric_protocol": "creator-standard-six-strata", "generation": protocol},
        "verifier_result": {"rewards": {metric: value, validity: float(valid)}},
        "measurement_review": {
            "version": "fgbench-creator-six-strata-v1",
            "implementation": {
                "import_path": "benchmarks.fgbench.projection:project",
                "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            "evidence": {
                "source_revision": deterministic.PINS["fgbench"],
                "scorer_sha256": actual,
                "parser": "standard",
                "prompt_variant": "campaign-zero-shot",
            },
            "metrics": {
                metric: {
                    "status": "valid",
                    "reason": reason,
                    "aggregation": "mean" if boolean else "sqrt_mean",
                    "value_semantics": "correctness"
                    if boolean
                    else "squared_error_or_null_when_creator_excludes_pair",
                },
                validity: {
                    "status": "valid",
                    "reason": "Creator-valid answer indicator over every observed row in this stratum",
                    "aggregation": "mean",
                },
            },
        },
        "fgbench_task_sha256": hashlib.sha256(
            json.dumps(task, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
