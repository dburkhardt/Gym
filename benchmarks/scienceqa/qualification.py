# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify unchanged caption-only inputs against inspected creator assets."""

import hashlib
import json
from pathlib import Path

from benchmarks.scientific_agent_campaign import deterministic


ASSETS = {
    "data/captions.json": "764e45f2b62b843f629941695af58d2adcd5613a25b5a74b1352ccace618537e",
    "data/scienceqa/problems.json": "4d9b598da966d9736dd79e430a97da861a2216aeb7483a5092350e823ab20ce7",
    "models/base_prompt.py": "22e7c551c891b4ebe5256ac8f9c1cfd7f75d34b2da71ba400e2036a360bc0e8b",
    "models/run_gpt3.py": "90341d4a99794a02ecf65ef2183584dd7d56958f3cef39729e150b49ba93be22",
}
FORMAT_SUFFIX = (
    "\n\nEnd your response with exactly: The answer is X.\nReplace X with the letter of your selected option."
)


def qualify(repo: Path, task: dict, details: dict) -> dict:
    """Return review evidence, never modify source inputs or creator scores."""
    hashes = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in ASSETS}
    matching = hashes == ASSETS
    meta = task["verifier_metadata"]
    variant = "unrecognized"
    matches_record = False
    if matching:
        original = json.loads((repo / "data/scienceqa/problems.json").read_text()).get(task["uuid"])
        captions = json.loads((repo / "data/captions.json").read_text())["captions"]
        if original is not None:
            original = {**original, "caption": captions.get(task["uuid"], "")}
            matches_record = all(meta.get(key) == value for key, value in original.items())
            base = deterministic.module(repo / "models/base_prompt.py")
            prompt = base.create_one_example(
                "QCM-A",
                original["question"],
                base.get_context_text(original, True),
                base.get_choice_text(original, list("ABCDE")),
                "",
                "",
                "",
                test_example=True,
            )
            inputs = task.get("responses_create_params", {}).get("input")
            for name, text in (
                ("caption-qcm-a-zero-shot", prompt),
                ("caption-qcm-a-explicit-format", prompt + FORMAT_SUFFIX),
            ):
                # Native parsing may add optional null fields to a message.
                if (
                    isinstance(inputs, list)
                    and len(inputs) == 1
                    and inputs[0].get("role") == "user"
                    and inputs[0].get("content") == text
                ):
                    variant = name
    recognized = matching and matches_record and variant != "unrecognized"
    caption_defect = recognized and task["uuid"] == "11434"
    common = (
        "Pinned source/prompt not matched"
        if not recognized
        else "Scientific correctness of the source item has not been reviewed"
    )
    strict_status = "invalid" if caption_defect else "unresolved"
    if caption_defect:
        common = "Pinned item11434 asks planetary volumes but its creator caption describes a baseball game; no image supplied"
    fallback = details["fallback_randomized"]
    return {
        "version": "scienceqa-caption-protocol-v1",
        "implementation": {
            "import_path": "benchmarks.scienceqa.qualification:qualify",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "evidence": {
            "source_revision": deterministic.PINS["scienceqa"],
            "scoring_adapter_sha256": hashlib.sha256(Path(deterministic.__file__).read_bytes()).hexdigest(),
            "asset_sha256": hashes,
            "task_sha256": hashlib.sha256(
                json.dumps(task, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "source_record_matches": matches_record,
            "protocol_variant": variant,
            "input_modality": "caption-only" if recognized else "unconfirmed",
            "subject": meta.get("subject"),
            "topic": meta.get("topic"),
            "category": meta.get("category"),
            "scope": "life-science-topic" if meta.get("topic") == "biology" else "not-established-as-life-sciences",
            "fallback_randomized": fallback,
            "fallback_seed": task["uuid"],
            "fallback_variant": "per-task seeded adaptation of creator random invalid-answer fallback",
        },
        "metrics": {
            "creator_seeded_fallback_accuracy": {
                "status": "invalid" if fallback or caption_defect else "unresolved",
                "reason": "Random fallback is not model answer correctness" if fallback else common,
            },
            "strict_accuracy": {"status": strict_status, "reason": common},
        },
    }
