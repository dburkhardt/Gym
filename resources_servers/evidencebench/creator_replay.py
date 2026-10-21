# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordered, source-bound creator scoring of saved answers; no generation or regeneration."""

import ast
import hashlib
import random
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np


PIN = "bf1d9633c694381c7b016fd56ee9f95f48593cc3"
PARSER = "parse_gpt_output"


def replay(rows: list[dict], *, creator_root: str, parser: str, seed: int = 42) -> dict:
    """Replay explicit GPT-parser adaptation with one ordered creator RNG stream."""
    if parser != PARSER:
        raise ValueError("Explicit parse_gpt_output adaptation required; Nemotron has no canonical creator mapping")
    root = Path(creator_root)
    if subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() != PIN:
        raise ValueError("Creator revision mismatch")
    sources = {}
    for name in ("postprocess_util.py", "evaluate.py", "parsing_first_round.py"):
        relative = f"Evaluation/post_process/{name}"
        data = (root / relative).read_bytes()
        if data != subprocess.check_output(["git", "-C", str(root), "show", f"{PIN}:{relative}"]):
            raise ValueError(f"Changed creator source: {relative}")
        sources[name] = data
    parser_tree = ast.parse(sources["postprocess_util.py"])
    node = next(n for n in parser_tree.body if isinstance(n, ast.FunctionDef) and n.name == parser)
    namespace = {"re": re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "pinned_evidencebench_parser", "exec"), namespace)
    parse = namespace[parser]
    dataset, predictions = {}, {}
    limit = None
    for row in rows:
        key = row["task_id"]
        if key in dataset:
            raise ValueError("Duplicate task identity")
        meta = row["verifier_metadata"]
        if limit is not None and meta["limit"] != limit:
            raise ValueError("Creator replay requires one limit per ordered batch")
        limit = meta["limit"]
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("This replay supports fixed positive limits only")
        dataset[key] = {"sentence_index2aspects": meta["sentence_index2aspects"]}
        predictions[key] = parse(row["prompt"], row["answer"])
    tree = ast.parse(sources["evaluate.py"])
    main = next(n for n in tree.body if isinstance(n, ast.If))
    loop = next(n for n in main.body if isinstance(n, ast.For))
    rng = random.Random(seed)
    before = hashlib.sha256(repr(rng.getstate()).encode()).hexdigest()
    scope = {
        "dataset": dataset,
        "parsed_list_dict": predictions,
        "sent_limit": limit,
        "cov_list": [],
        "random": rng,
        "np": np,
        "warnings": warnings,
        "tqdm": lambda x: x,
    }
    # Execute the unchanged creator loop, including its errors and 100000 shuffles.
    exec(compile(ast.Module(body=[loop], type_ignores=[]), "pinned_evidencebench_evaluate", "exec"), scope)
    if len(scope["cov_list"]) != len(rows):
        raise ValueError("Creator skipped a reference; refuse ambiguous row-to-score mapping")
    return {
        "protocol": "creator-gpt-parser-saved-answer-adaptation-v1",
        "creator_commit": PIN,
        "source_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in sources.items()},
        "parser": parser,
        "seed": seed,
        "python": sys.version,
        "numpy": np.__version__,
        "rng_before": before,
        "rng_after": hashlib.sha256(repr(rng.getstate()).encode()).hexdigest(),
        "task_order": list(dataset),
        "scores": scope["cov_list"],
        "mean": float(np.mean(scope["cov_list"])) if rows else None,
        "qualification": "Explicit parser adaptation; saved answers only, no creator regeneration; ordered subset RNG is not full-population RNG",
    }


def main() -> None:
    """Project saved native results without changing original records."""
    import argparse
    import json

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", required=True)
    cli.add_argument("--output", required=True)
    cli.add_argument("--creator-root", required=True)
    cli.add_argument("--parser", required=True, choices=[PARSER])
    cli.add_argument("--seed", type=int, default=42)
    args = cli.parse_args()
    source = Path(args.input).read_bytes()
    records = [json.loads(line) for line in source.splitlines() if line.strip()]
    rows = []
    scored_indices = []
    for index, record in enumerate(records):
        response = record.get("response")
        if not isinstance(response, dict) or not isinstance(response.get("output"), list):
            continue
        scored_indices.append(index)
        answer = "".join(
            content["text"]
            for item in response["output"]
            if item.get("type") == "message"
            for content in item.get("content", [])
            if content.get("type") == "output_text"
        )
        rows.append(
            {
                "task_id": record["verifier_metadata"]["task_id"],
                "verifier_metadata": record["verifier_metadata"],
                "prompt": "",
                "answer": answer,
            }
        )
    result = replay(rows, creator_root=args.creator_root, parser=args.parser, seed=args.seed)
    result["input_sha256"] = hashlib.sha256(source).hexdigest()
    result["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["original_rewards"] = [record.get("reward") for record in records]
    result["original_masks"] = [record.get("mask_sample") for record in records]
    result["original_failure_kinds"] = [record.get("failure_kind") for record in records]
    projected = [None] * len(records)
    for index, score in zip(scored_indices, result["scores"]):
        projected[index] = score
    result["record_scores"] = projected
    result["deltas"] = [
        score - record["reward"]
        if score is not None
        and isinstance(record.get("reward"), (int, float))
        and not isinstance(record.get("reward"), bool)
        else None
        for score, record in zip(projected, records)
    ]
    result["records"] = [
        {
            "input_ordinal": index,
            "record_sha256": hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
            "identity": {
                key: record.get(key)
                for key in (
                    "task_id",
                    "_ng_task_index",
                    "_ng_rollout_index",
                    "_ng_attempt_index",
                    "agent_ref",
                    "task_source",
                )
            },
            "verifier_task_id": (record.get("verifier_metadata") or {}).get("task_id"),
            "failure_kind": record.get("failure_kind"),
            "failure_reason": record.get("failure_reason"),
            "mask_sample": record.get("mask_sample"),
            "projection_disposition": "raw_creator_score_only"
            if index in scored_indices
            else "unscored_missing_response",
        }
        for index, record in enumerate(records)
    ]
    with Path(args.output).open("x") as output:
        json.dump(result, output, indent=2)


if __name__ == "__main__":
    main()
