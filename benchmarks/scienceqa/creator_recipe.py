# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned creator recipe preparation, not a chat substitution or runnable adapter."""

import hashlib
import json
import random
import re
from pathlib import Path
from types import SimpleNamespace

from benchmarks.scienceqa.qualification import ASSETS
from benchmarks.scientific_agent_campaign.deterministic import PINS, module, validate_repo


PROTOCOL = "scienceqa-creator-qcm-ale-2shot-no-caption-completion-v1"
SPLIT_SHA256 = "85c7820b9f136e572579a01dab2c77a7f8140b59211e421fa5182491d45ee3a7"


def prepare(repo: Path, task_ids: list[str]) -> dict:
    """Keep solver requests separate from reference-bearing evaluator records."""
    validate_repo("scienceqa", repo)
    hashes = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in ASSETS}
    if (
        hashes != ASSETS
        or hashlib.sha256((repo / "data/scienceqa/pid_splits.json").read_bytes()).hexdigest() != SPLIT_SHA256
    ):
        raise ValueError("Creator recipe assets do not match the pinned source")
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("Task IDs must be nonempty and unique")
    problems = json.loads((repo / "data/scienceqa/problems.json").read_text())
    splits = json.loads((repo / "data/scienceqa/pid_splits.json").read_text())
    if any(task_id not in splits["test"] for task_id in task_ids):
        raise ValueError("Creator recipe task is outside the pinned test split")
    rng = random.Random(3)
    shots = rng.sample(splits["train"], 2)
    args = SimpleNamespace(prompt_format="QCM-ALE", use_caption=False, options=list("ABCDE"))
    creator = module(repo / "models/base_prompt.py")
    requests, references = [], []
    for task_id in task_ids:
        prompt = creator.build_prompt(problems, shots, task_id, args)
        request = {
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": 512,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": ["\n"],
        }
        requests.append({"task_id": task_id, "protocol": PROTOCOL, "request": request})
        references.append(
            {"task_id": task_id, "answer": problems[task_id]["answer"], "choices": problems[task_id]["choices"]}
        )
    return {
        "protocol": PROTOCOL,
        "source_revision": PINS["scienceqa"],
        "source_hashes": hashes | {"data/scienceqa/pid_splits.json": SPLIT_SHA256},
        "shot_ids": shots,
        "seed": 3,
        "creator_engine": "text-davinci-002",
        "execution_capability_required": "raw_text_completion",
        "execution_status": "prepared_native_route_requires_provider_qualification",
        "solver_requests": requests,
        "evaluator_references": references,
        "rng_scope": "Fresh seed3, consume two-shot selection, then fallback draws in this explicit task order; not full-test RNG parity for a subset",
    }


def score_ordered(repo: Path, packet: dict, outputs: list[str]) -> list[dict]:
    """Preserve creator fallback stream for this explicitly ordered subset."""
    ids = [row["task_id"] for row in packet["solver_requests"]]
    expected = prepare(repo, ids)
    comparison = dict(packet)
    if comparison.get("execution_status") == "blocked_missing_native_guarded_raw_completion_route":
        # Historical metadata only; requests/references/source/RNG still compare exactly.
        comparison["execution_status"] = expected["execution_status"]
    if comparison != expected:
        raise ValueError("Creator recipe packet differs from pinned preparation")
    if len(outputs) != len(ids) or any(not isinstance(text, str) for text in outputs):
        raise ValueError("Provide exactly one string completion for each ordered task")
    splits = json.loads((repo / "data/scienceqa/pid_splits.json").read_text())
    rng = random.Random(3)
    rng.sample(splits["train"], 2)
    results = []
    for reference, text in zip(packet["evaluator_references"], outputs):
        found = re.findall(r"The answer is ([A-Z]).", text.strip())
        prediction = found[0] if len(found) == 1 else "FAILED"
        valid = prediction in list("ABCDE")[: len(reference["choices"])]
        index = "ABCDE".index(prediction) if valid else rng.choice(range(len(reference["choices"])))
        results.append(
            {
                "task_id": reference["task_id"],
                "creator_raw_accuracy": float(index == reference["answer"]),
                "strict_accuracy": float(valid and index == reference["answer"]),
                "fallback_randomized": not valid,
                "predicted_index": index,
                "measurement_valid": False if not valid else None,
                "qualification": "Random fallback is not answer correctness"
                if not valid
                else "Reference correctness not independently reviewed",
            }
        )
    return results
