# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage frozen campaign rows through Gym without resampling or changing prompts."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from nemo_gym.dataset_staging import stage_file


def prepare_sample(
    destination: Path,
    *,
    source: Path | None = None,
    expected_source_sha256: str | None = None,
    identity_metadata_key: str = "task_id",
) -> Path:
    """Preserve every source field and expose its frozen identity to Gym's planner.

    The sibling provenance receipt binds the original bytes and normalized output.
    Consumers must compare its output hash before using a receipt after interruption.
    """
    source = source or Path(os.environ["CAMPAIGN_SAMPLE_JSONL"])
    expected_source_sha256 = expected_source_sha256 or os.environ.get("CAMPAIGN_SAMPLE_SHA256")
    content = source.read_bytes()
    source_hash = hashlib.sha256(content).hexdigest()
    if expected_source_sha256 is not None and source_hash != expected_source_sha256:
        raise ValueError("Frozen campaign sample SHA-256 mismatch")
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    if not rows or len(rows) > 50:
        raise ValueError("Expected 1-50 preselected sample rows")
    ids = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Expected JSON objects for frozen sample rows")
        metadata = row.get("verifier_metadata")
        if not isinstance(metadata, dict) or not row.get("responses_create_params", {}).get("input"):
            raise ValueError("Missing solver input or isolated verifier metadata")
        identity = row.get("task_id", row.get("uuid", metadata.get(identity_metadata_key)))
        if isinstance(identity, bool) or not isinstance(identity, (str, int)) or not str(identity):
            raise ValueError("Missing stable task identity; positional IDs are not allowed")
        row.setdefault("task_id", str(identity))
        ids.append(str(identity))
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate frozen sample IDs")
    output = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
    output_hash = hashlib.sha256(output).hexdigest()
    # Resolve the trusted parent without following the final publication path.
    # stage_file must see and reject a destination symlink itself.
    destination = destination.parent.resolve() / destination.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"source_sha256": source_hash, "output_sha256": output_hash, "task_ids": ids, "rows": len(rows)}
    with tempfile.TemporaryDirectory(prefix="gym-frozen-sample-") as temporary:
        candidate = Path(temporary) / "sample.jsonl"
        candidate.write_bytes(output)
        stage_file(candidate, destination, expected_sha256=output_hash)
        proof = Path(temporary) / "provenance.json"
        proof.write_text(json.dumps(receipt, indent=2) + "\n")
        stage_file(proof, destination.with_suffix(".source.json"))
    return destination
