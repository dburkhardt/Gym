# SPDX-License-Identifier: Apache-2.0
"""Regrade saved BIABench outputs with the pinned creator outcome scorer.

Execute in a separate verifier process/environment; never in the solver context.
Process-checklist scoring is distinct and is not performed by this command.
"""

import argparse
import hashlib
import importlib
import inspect
import json
import sys
from pathlib import Path

from benchmarks.biabench.prepare import validate


def regrade(source: Path, task: Path, predictions: Path, output: Path) -> dict:
    source, task, predictions = source.resolve(), task.resolve(), predictions.resolve()
    if output.exists():
        raise FileExistsError("Choose a new scorer result path; historical results are immutable")
    if not predictions.is_dir():
        raise ValueError("Saved prediction directory is required")
    manifest_path = Path(__file__).with_name("frozen_sources.json")
    manifest = json.loads(manifest_path.read_text())
    validate(source, manifest["creator_files"])
    validate(task, manifest["task_files"])
    sys.path.insert(0, str(source))
    module = importlib.import_module("bioimage_agent_bench.evaluators.wound_healing_kymograph")
    actual = Path(inspect.getfile(module)).resolve()
    expected = source / "bioimage_agent_bench/evaluators/wound_healing_kymograph.py"
    if actual != expected:
        raise ValueError(f"Wrong creator scorer imported: {actual}")
    import yaml

    rubric = yaml.safe_load((task / "evaluation_rubric.yaml").read_text())
    metrics = module.compute_wound_healing_kymograph_metrics(
        predictions, task / "evaluation", rubric.get("metric_config", {}), task
    )
    record = {
        "task_id": manifest["task_id"],
        "outcome_metrics": metrics,
        "process_score": None,
        "process_status": "not_evaluated",
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "scorer_sha256": hashlib.sha256(actual.read_bytes()).hexdigest(),
        "predictions": str(predictions),
        "new_model_calls": 0,
        "qualification": "Creator outcome metrics only; not a complete benchmark score or process-checklist result.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "task", "predictions", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    regrade(args.source, args.task, args.predictions, args.output)
