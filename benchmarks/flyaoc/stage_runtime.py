# SPDX-License-Identifier: Apache-2.0
"""Publish a pinned solver-only source/corpus/ontology view; never copy references."""

import json
from pathlib import Path

from nemo_gym.dataset_staging import stage_file


def stage_runtime(*, source: Path, corpus: Path, ontologies: Path, destination: Path) -> Path:
    runtime = json.loads(Path(__file__).with_name("runtime_sources.json").read_text())
    frozen = json.loads(Path(__file__).with_name("frozen_sources.json").read_text())
    for name, sha in runtime["source_files"].items():
        stage_file(source / name, destination / "src" / name, expected_sha256=sha)
    for name, sha in runtime["corpus_files"].items():
        stage_file(corpus / name, destination / "cache/corpus" / name, expected_sha256=sha)
    for name, sha in frozen["ontology_sha256"].items():
        stage_file(ontologies / name, destination / "ontologies" / name, expected_sha256=sha)
        stage_file(
            ontologies / name,
            destination
            / "cache/hf/datasets--anonymous-042--flyaoc/snapshots"
            / runtime["dataset_revision"]
            / "ontologies"
            / name,
            expected_sha256=sha,
        )
    return destination
