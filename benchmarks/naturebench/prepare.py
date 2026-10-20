# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage the preserved NatureBench selection without downloading or substituting tasks."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_gym.dataset_staging import stage_file


def prepare() -> Path:
    source = Path(os.environ["NATUREBENCH_FROZEN_SAMPLE"])
    content = source.read_bytes()
    source_hash = hashlib.sha256(content).hexdigest()
    expected = os.environ.get("NATUREBENCH_SAMPLE_SHA256")
    if expected and expected != source_hash:
        raise ValueError("NatureBench frozen source checksum mismatch")
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    if not rows or len(rows) > 50:
        raise ValueError("Expected 1-50 preserved NatureBench rows")
    ids = []
    for row in rows:
        identity = row.get("instance_id")
        if not isinstance(identity, str) or not identity.startswith("naturebench::"):
            raise ValueError("Missing NatureBench instance_id")
        paper = identity.removeprefix("naturebench::")
        if not paper or "/" in paper or "\\" in paper or paper in (".", ".."):
            raise ValueError("Unsafe NatureBench task identity")
        if row.get("task_id", identity) != identity:
            raise ValueError("Conflicting task identity")
        if row.get("responses_create_params", {}).get("input") != []:
            raise ValueError("Harbor task instruction must remain in its pinned task package")
        row.setdefault("task_id", identity)
        ids.append(identity)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate NatureBench task IDs")
    if archives := os.environ.get("NATUREBENCH_ARCHIVES_DIR"):
        from benchmarks.naturebench.prepare_runtime import materialize_archive

        manifest = json.loads((Path(__file__).parent / "frozen_sources.json").read_text())
        for identity in ids:
            paper = identity.removeprefix("naturebench::")
            if paper not in manifest["tasks"]:
                raise ValueError("No pinned archive manifest for selected task")
            materialize_archive(
                Path(archives) / (paper + ".tar.gz"),
                Path(__file__).parent / "data" / "tasks" / paper,
                expected_sha256=manifest["tasks"][paper]["archive_sha256"],
            )
    output = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    output_hash = hashlib.sha256(output).hexdigest()
    destination = Path(__file__).parent / "data" / "benchmark.jsonl"
    receipt = dict(
        source_sha256=source_hash,
        output_sha256=output_hash,
        task_ids=ids,
        normalization="instance_id alias to task_id",
        execution_readiness="Task archives, GPU contracts and guarded judge require separate validation",
    )
    with TemporaryDirectory() as directory:
        candidate = Path(directory) / "sample.jsonl"
        candidate.write_bytes(output)
        stage_file(candidate, destination, expected_sha256=output_hash)
        candidate.write_text(json.dumps(receipt, indent=2) + "\n")
        stage_file(candidate, destination.with_suffix(".source.json"))
    return destination


if __name__ == "__main__":
    print(prepare())
