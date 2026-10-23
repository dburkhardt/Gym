# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate an exact creator EVAL export; preserve review collections and qrels."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from resources_servers.csmed_ranking.grading import read_json, validate_rankings


RETRIEVAL_SHA256 = "8fdf75f443528ac5c7dbac62a592969b2d51cc6b89539d8fd5b45b0f971d5ace"


def corpus_occurrences(collection, documents):
    """Validate raw ordered corpus, keeping occurrence identity separate from PMID."""
    if not isinstance(collection, list) or not collection or len(collection) != len(documents):
        raise ValueError("Nonempty aligned creator corpus required")
    labels, occurrences = {}, []
    for index, (doc, raw) in enumerate(zip(collection, documents)):
        identifier, text = doc.get("id"), doc.get("text")
        values = raw.get("labels")
        if not isinstance(identifier, str) or not identifier or not isinstance(text, str) or not text.strip():
            raise ValueError("Nonempty document ID and text required")
        if not isinstance(values, list) or len(values) != 1 or type(values[0]) is not int or values[0] not in (0, 1):
            raise ValueError("Exactly one binary integer label per document required")
        if identifier in labels and labels[identifier] != values[0]:
            raise ValueError("Conflicting labels for repeated document ID")
        labels[identifier] = values[0]
        occurrences.append(
            {
                "occurrence_id": index,
                "document_id": identifier,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    return labels, occurrences


def prepare(source, eval_export, expected_sha256, destination, preserve_occurrences=False):
    source, eval_export = Path(source), Path(eval_export)
    if hashlib.sha256(source.read_bytes()).hexdigest() != RETRIEVAL_SHA256:
        raise ValueError("Creator retrieval source changed")
    if hashlib.sha256(eval_export.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Creator EVAL export checksum changed")
    data = read_json(eval_export)
    reviews = data.get("EVAL")
    if not isinstance(reviews, dict) or not reviews:
        raise ValueError("Explicit nonempty creator EVAL review mapping required")
    tree = ast.parse(source.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"extract_review_details", "prepare_data"}
    ]
    if len(nodes) != 2:
        raise ValueError("Creator preparation functions missing")
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    inputs, qrels, occurrence_maps = {}, {}, {}
    for review, body in reviews.items():
        try:
            queries = namespace["extract_review_details"](body)
            collection, labels = namespace["prepare_data"](body)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError("Incomplete creator review/document fields") from error
        if (
            not isinstance(review, str)
            or not review
            or any(not isinstance(v, str) or not v.strip() for v in queries.values())
        ):
            raise ValueError("Complete creator query text required")
        checked_labels, occurrences = corpus_occurrences(collection, body["data"]["train"])
        if labels != checked_labels:
            raise ValueError("Creator qrels projection differs from validated labels")
        if not preserve_occurrences and len(collection) != len(labels):
            raise ValueError("Complete unique document IDs/text required")
        # Ranking remains one score per distinct PMID. Raw rows are never deduplicated.
        validate_rankings({review: [[doc, 0] for doc in labels]}, {review: labels})
        occurrence_maps[review] = occurrences
        inputs[review] = {"queries": queries, "collection": collection}
        qrels[review] = labels
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "review-inputs.json": inputs,
        "qrels.json": qrels,
        "preparation.json": {
            "source_sha256": RETRIEVAL_SHA256,
            "eval_export_sha256": expected_sha256,
            "review_ids": list(inputs),
            "document_counts": {q: len(v) for q, v in qrels.items()},
            "preserve_occurrences": preserve_occurrences,
            "occurrence_counts": {q: len(v) for q, v in occurrence_maps.items()},
            "qualification": "caller-supplied complete EVAL export; no authoritative corpus provenance or solver qualification",
        },
    }
    if preserve_occurrences:
        artifacts["corpus-occurrences.json"] = occurrence_maps
    for name, value in artifacts.items():
        (destination / name).write_text(json.dumps(value, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "eval-export", "destination"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--preserve-occurrences", action="store_true")
    print(prepare(**vars(parser.parse_args())))
