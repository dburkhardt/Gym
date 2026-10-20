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


def prediction_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Prediction artifacts must not be symlinks: {path}")
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[str(path.relative_to(root))] = digest.hexdigest()
    return hashes


def imported_creator_hashes(source: Path, pinned: dict[str, str]) -> dict[str, str]:
    hashes = {}
    for name, module in tuple(sys.modules.items()):
        if name != "bioimage_agent_bench" and not name.startswith("bioimage_agent_bench."):
            continue
        file = getattr(module, "__file__", None)
        if file is None:
            continue
        path = Path(file).resolve()
        if not path.is_relative_to(source):
            raise ValueError(f"Mixed creator source import: {name}")
        relative = str(path.relative_to(source))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if pinned.get(relative) != actual:
            raise ValueError(f"Unpinned creator import: {name}")
        hashes[relative] = actual
    return hashes


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
    imported_creator_hashes(source, manifest["creator_files"])
    before = prediction_hashes(predictions)
    import yaml

    rubric = yaml.safe_load((task / "evaluation_rubric.yaml").read_text())
    metrics = module.compute_wound_healing_kymograph_metrics(
        predictions, task / "evaluation", rubric.get("metric_config", {}), task
    )
    imports = imported_creator_hashes(source, manifest["creator_files"])
    if prediction_hashes(predictions) != before:
        raise ValueError("Scorer mutated saved prediction artifacts")
    record = {
        "task_id": manifest["task_id"],
        "outcome_metrics": metrics,
        "process_score": None,
        "process_status": "not_evaluated",
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "scorer_sha256": hashlib.sha256(actual.read_bytes()).hexdigest(),
        "predictions": str(predictions),
        "prediction_sha256": before,
        "creator_import_sha256": imports,
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
