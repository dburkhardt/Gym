# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned creator image-baseline inputs and unchanged magnetic multiset scorer."""

import ast
import hashlib
import json
from pathlib import Path


MANIFEST = Path(__file__).with_name("frozen.json")


class ModelOutputError(ValueError):
    """The model output cannot be consumed by the unchanged creator JSON protocol."""


def validate_source(source: Path) -> dict:
    """Bind only source files actually used, refusing modified or symlinked inputs."""
    manifest = json.loads(MANIFEST.read_text())
    for name, digest in manifest["source_sha256"].items():
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError(f"Unsafe creator source: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Creator source hash mismatch: {name}")
    return manifest


def creator_definitions(source: Path) -> dict:
    """Load pinned constants/prompts and exact scorer helpers without network imports."""
    validate_source(source)
    namespace = {}
    for relative in ("LLM/src/constants.py", "LLM/data/prompts/magnetic.py"):
        tree = ast.parse((source / relative).read_text())
        statements = [node for node in tree.body if isinstance(node, ast.Assign)]
        exec(compile(ast.Module(body=statements, type_ignores=[]), relative, "exec"), namespace)
    return namespace


def score(source: Path, *, article_id: str, answer: str) -> dict:
    """Keep creator numeric normalization, duplicate handling and column multisets."""
    import pandas as pd

    manifest = validate_source(source)
    article = next((a for a in manifest["articles"] if a["id"] == article_id), None)
    if article is None:
        raise ValueError("Article is not in the explicit two-article sample")
    definitions = creator_definitions(source)
    columns = definitions["EXTRACTED_COLUMNS"]["magnetic"]
    numeric = definitions["NUMERIC_COLUMNS"]["magnetic"]
    tree = ast.parse((source / "LLM/src/metric_calc.py").read_text())
    functions = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"convert_comma", "calc_metrics"}
    ]
    namespace = {"pd": pd}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "creator_metric_calc", "exec"), namespace)
    gold = pd.read_csv(source / "datasets/Nanomag.csv")
    for column in numeric:
        gold[column] = gold[column].apply(namespace["convert_comma"])
    gold = gold.fillna("NOT_DETECTED")
    gold = gold.loc[(gold["access"] == 1) & (gold["pdf"].str.lower() == article_id.lower()), columns]
    if len(gold) != article["gold_rows"]:
        raise ValueError("Frozen article gold row count changed")
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError as exc:
        raise ModelOutputError("Model output is not valid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ModelOutputError("Creator output requires a JSON samples array")
    samples = payload["samples"]
    if not all(isinstance(row, dict) for row in samples):
        raise ModelOutputError("Creator samples must be objects")
    predicted = pd.DataFrame([{c: row.get(c) for c in columns} for row in samples], columns=columns).drop_duplicates()
    metrics = namespace["calc_metrics"](gold, predicted)
    return {
        "reward": float(metrics["f1"].mean()),
        "creator_column_metrics": metrics.to_dict(orient="index"),
        "metric": "mean_column_f1_summary_of_creator_metrics",
        "protocol": manifest["version"],
        "source_revision": manifest["source_revision"],
        "gold_rows": len(gold),
        "predicted_rows_after_deduplication": len(predicted),
        "qualification": "Column multisets ignore row associations; scalar is an explicit summary, not a published suite score",
    }
