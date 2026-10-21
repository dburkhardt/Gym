# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Complete-review ranking validation and unchanged creator evaluate_runs.

No scalar accuracy/reward or LLM solver is invented. Output is the creator's21
ranx metrics and supplementary measures, with complete review denominators.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path


SOURCE_SHA256 = "9911926a1bee551750f94e81a61201691a7d06ec4351f9ee1734e9cd09107f19"
SOURCE_REVISION = "81cd35498bdea8981e5310842c32398625a10e95"
RANX_VERSION = "0.3.18"


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON identity: {key}")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def validate_rankings(rankings, qrels):
    if not isinstance(rankings, dict) or not isinstance(qrels, dict) or not qrels or set(rankings) != set(qrels):
        raise ValueError("Exact nonempty review population required")
    run = {}
    for query, labels in qrels.items():
        if not isinstance(query, str) or not query or not isinstance(labels, dict) or not labels:
            raise ValueError("Nonempty string review IDs and qrels required")
        if any(not isinstance(doc, str) or not doc for doc in labels):
            raise ValueError("Nonempty string document IDs required")
        if any(type(label) is not int or label not in (0, 1) for label in labels.values()):
            raise ValueError("Binary integer qrels required")
        if not 0 < sum(labels.values()) < len(labels):
            raise ValueError("Creator normalized measures require both relevance classes")
        pairs = rankings[query]
        if not isinstance(pairs, list) or any(not isinstance(pair, (list, tuple)) or len(pair) != 2 for pair in pairs):
            raise ValueError("Ordered document/score pairs required")
        ids = [pair[0] for pair in pairs]
        if any(not isinstance(doc, str) for doc in ids) or len(ids) != len(set(ids)) or set(ids) != set(labels):
            raise ValueError("Complete unique corpus ranking required")
        values = [pair[1] for pair in pairs]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError("Finite numeric scores required")
        if any(a < b for a, b in zip(values, values[1:])):
            raise ValueError("Scores must already descend; preserve source tie order")
        run[query] = dict(pairs)
    return run


def creator(source):
    source = Path(source)
    if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Creator measures source changed")
    if importlib.metadata.version("ranx") != RANX_VERSION:
        raise ValueError("Creator metric control requires ranx0.3.18")
    spec = importlib.util.spec_from_file_location("_csmed_bound_measures", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_runs(source, rankings, qrels, destination):
    if not isinstance(rankings, dict) or not rankings:
        raise ValueError("Nonempty named runs required")
    runs = {}
    for name, run in rankings.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Nonempty run name required")
        runs[name] = validate_rankings(run, qrels)
    module = creator(source)
    captured = []
    original_compare = module.compare

    def capture_compare(*args, **kwargs):
        report = original_compare(*args, **kwargs)
        captured.append(report)
        return report

    module.compare = capture_compare
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    # The creator emits its unmodified LaTeX/comparison and supplementary report.
    module.evaluate_runs(runs, qrels, outfile_path=str(destination))
    if len(captured) != 1:
        raise ValueError("Unexpected creator reporting behavior")
    report = captured[0]
    result = {
        "source_revision": SOURCE_REVISION,
        "rankings_sha256": hashlib.sha256(
            json.dumps(rankings, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "qrels_sha256": hashlib.sha256(json.dumps(qrels, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "source_sha256": SOURCE_SHA256,
        "ranx_version": RANX_VERSION,
        "review_count": len(qrels),
        "document_counts": {q: len(docs) for q, docs in qrels.items()},
        "metrics": report.results,
        "metric_names": report.metrics,
        "supplementary": read_json(destination / "other_measures.json"),
        "scalar_reward": None,
        "qualification": "complete-review ranking verifier; no solver or dataset access acceptance",
    }
    (destination / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "rankings", "qrels", "destination"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    score_runs(args.source, read_json(args.rankings), read_json(args.qrels), args.destination)
