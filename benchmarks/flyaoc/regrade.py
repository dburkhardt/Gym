# SPDX-License-Identifier: Apache-2.0
"""Recompute native FlyAOC metrics from saved predictions, without inference/downloads."""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regrade(*, source: Path, references: Path, ontologies: Path, predictions: Path, output: Path) -> dict:
    manifest_path = Path(__file__).with_name("frozen_sources.json")
    manifest = json.loads(manifest_path.read_text())
    source = source.resolve(strict=True)
    if output.exists():
        raise FileExistsError("Choose a new scorer output path")
    for name, checksum in manifest["scorer_source_files"].items():
        path = source / name
        if path.is_symlink() or sha(path) != checksum:
            raise ValueError(f"FlyAOC scorer source mismatch: {name}")
    if sha(references) != manifest["reference_records_sha256"]:
        raise ValueError("FlyAOC reference checksum mismatch")
    for name, checksum in manifest["ontology_sha256"].items():
        if sha(ontologies / name) != checksum:
            raise ValueError(f"FlyAOC ontology checksum mismatch: {name}")
    before = sha(predictions)
    predicted = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    ids = [row["gene_id"] for row in predicted]
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(manifest["frozen_task_ids"]):
        raise ValueError("Predictions must contain unique frozen gene IDs")
    refs = {row["gene_id"]: row for row in map(json.loads, references.read_text().splitlines())}
    sys.path.insert(0, str(source))
    native = importlib.import_module("flyaoc.evaluation.evaluate")

    def imported_sources():
        result = {}
        for name, module in tuple(sys.modules.items()):
            if name != "flyaoc" and not name.startswith("flyaoc."):
                continue
            file = getattr(module, "__file__", None)
            if not file:
                continue
            path = Path(file).resolve()
            if not path.is_relative_to(source):
                raise ValueError(f"Mixed FlyAOC scorer import: {name}")
            relative = str(path.relative_to(source))
            digest = sha(path)
            if manifest["scorer_source_files"].get(relative) != digest:
                raise ValueError(f"Unpinned FlyAOC scorer import: {name}")
            result[relative] = digest
        return result

    imported_sources()
    go = native.GoSimilarity(ontologies / "go-basic.obo")
    anatomy = native.AnatomySimilarity(ontologies / "fly_anatomy.obo")
    config = native.EvaluationConfig()
    genes = [
        native.evaluate_gene(
            prediction=row, benchmark_row=refs[row["gene_id"]], config=config, go_sim=go, anatomy_sim=anatomy
        )
        for row in predicted
    ]
    result = {"n_genes": len(genes), "aggregate": native.aggregate_gene_results(genes), "genes": genes}
    if sha(predictions) != before:
        raise ValueError("Saved predictions changed during regrade")
    receipt = {
        "evaluation": result,
        "prediction_sha256": before,
        "source_manifest_sha256": sha(manifest_path),
        "imported_source_sha256": imported_sources(),
        "new_model_calls": 0,
        "scope": "Creator ontology-backed metrics; factual correctness remains unadjudicated",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "references", "ontologies", "predictions", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    regrade(**vars(parser.parse_args()))
