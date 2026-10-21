# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind preserved native fixed-pool RAG prompts, never corpus-search prompts."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from nemo_gym.dataset_staging import stage_file


def stage(materialized_root: Path, destination: Path, count: int = 20) -> Path:
    """Preserve native pilot5 or pilot5+remaining15 rows and verify all prompt bytes."""
    if count not in (5, 20):
        raise ValueError("MetaSyn native frozen sample must contain5 or20 tasks")
    manifest = json.loads(Path(__file__).with_name("frozen-samples.json").read_text())
    preparation_hash = hashlib.sha256((materialized_root / "rag-preparation.json").read_bytes()).hexdigest()
    if preparation_hash != manifest["preparation_sha256"]:
        raise ValueError("MetaSyn native retrieval preparation provenance mismatch")
    specs = manifest["samples"]
    rows = []
    sources = []
    prompts = []
    for spec in specs[:1] if count == 5 else specs:
        source = materialized_root / spec["source_file"]
        data = source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != spec["sha256"]:
            raise ValueError("MetaSyn frozen native source checksum mismatch")
        source_rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        if len(source_rows) != len(spec["rows"]):
            raise ValueError("MetaSyn frozen row count mismatch")
        for row, expected in zip(source_rows, spec["rows"], strict=True):
            review_id = row["verifier_metadata"]["review_id"]
            if isinstance(review_id, bool) or review_id != expected["review_id"]:
                raise ValueError("MetaSyn frozen review identity mismatch")
            prompt_path = materialized_root / "rag" / str(review_id) / "prompt.txt"
            prompt_bytes = prompt_path.read_bytes()
            if hashlib.sha256(prompt_bytes).hexdigest() != expected["prompt_sha256"]:
                raise ValueError("MetaSyn native RAG prompt checksum mismatch")
            prompt = prompt_path.read_text()
            if row["responses_create_params"]["input"] != [{"role": "user", "content": prompt}]:
                raise ValueError("MetaSyn row differs from native fixed-pool RAG prompt")
            identity = f"metasyn::{review_id}"
            if row.get("instance_id", identity) != identity or row.get("task_id", identity) != identity:
                raise ValueError("MetaSyn stable identity mismatch")
            if row.get("agent_ref", {}).get("name") != "metasyn_agent":
                raise ValueError("MetaSyn native agent routing mismatch")
            row["task_id"] = identity
            rows.append(row)
            prompts.append({"task_id": identity, "prompt_sha256": expected["prompt_sha256"]})
        sources.append({"path": spec["source_file"], "sha256": digest, "rows": len(source_rows)})
    ids = [r["task_id"] for r in rows]
    if len(ids) != len(set(ids)) or len(ids) != count:
        raise ValueError("MetaSyn duplicate or missing frozen task IDs")
    output = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
    output_hash = hashlib.sha256(output).hexdigest()
    destination = destination.parent.resolve() / destination.name
    receipt = {
        "source_revision": "c8fa07d89c44093d623f9a213c6bf070f40ab960",
        "protocol": "native BM25 top200 fixed-pool RAG; Nemotron generator/judge variant",
        "retrieval_preparation_sha256": preparation_hash,
        "sources": sources,
        "prompts": prompts,
        "output_sha256": output_hash,
        "task_ids": ids,
        "rows": count,
        "normalizations": ["task_id from immutable review_id; all source fields retained"],
    }
    with tempfile.TemporaryDirectory(prefix="gym-metasyn-sample-") as temporary:
        data_path = Path(temporary) / "benchmark.jsonl"
        data_path.write_bytes(output)
        stage_file(data_path, destination, expected_sha256=output_hash)
        proof = Path(temporary) / "provenance.json"
        proof.write_text(json.dumps(receipt, indent=2) + "\n")
        stage_file(proof, destination.with_suffix(".source.json"))
    return destination


def prepare() -> Path:
    """Prepare native frozen rows for Gym without invoking the external runner."""
    return stage(
        Path(os.environ["METASYN_DATA"]),
        Path(os.environ.get("METASYN_PREPARED_SAMPLE", "benchmarks/metasyn/data/benchmark.jsonl")),
        int(os.environ.get("METASYN_SAMPLE_COUNT", "20")),
    )


if __name__ == "__main__":
    print(prepare())
