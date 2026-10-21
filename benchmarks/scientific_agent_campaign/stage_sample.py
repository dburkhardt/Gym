# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage frozen campaign rows through Gym without resampling or changing prompts."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from nemo_gym.dataset_staging import stage_file


def is_sample_current(
    fpath: Path,
    *,
    default_destination: Path,
    source: Path | None = None,
    expected_normalizations: list[str] | None = None,
) -> bool:
    """Check cached publication against its receipt and currently selected source.

    Explicit external prepared files need no campaign receipt. If a frozen source
    is selected, an unreceipted external file must be that exact source. Invalid
    overrides are refused, never rewritten by the wrapper's canonical prepare().
    """
    destination = Path(fpath)
    external = (destination.parent.resolve() / destination.name) != (
        default_destination.parent.resolve() / default_destination.name
    )
    receipt_path = destination.with_suffix(".source.json")
    source_name = source or os.environ.get("CAMPAIGN_SAMPLE_JSONL")
    expected = os.environ.get("CAMPAIGN_SAMPLE_SHA256")
    source_hash = None
    if source_name:
        source_hash = hashlib.sha256(Path(source_name).read_bytes()).hexdigest()
        if expected is not None and source_hash != expected:
            raise ValueError("Frozen campaign sample SHA-256 mismatch")
    elif expected is not None:
        source_hash = expected
    if destination.is_symlink() or receipt_path.is_symlink():
        raise ValueError("Campaign cached sample and receipt must not be symlinks")
    content = destination.read_bytes()
    output_hash = hashlib.sha256(content).hexdigest()
    if external and not receipt_path.exists():
        if source_hash is None or output_hash == source_hash:
            return True
        raise ValueError("External prepared sample does not match the selected frozen source; it was not modified")
    try:
        receipt = json.loads(receipt_path.read_text())
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        ids = [str(row["task_id"]) for row in rows]
        valid = (
            isinstance(receipt, dict)
            and receipt.get("output_sha256") == output_hash
            and isinstance(receipt.get("source_sha256"), str)
            and len(receipt["source_sha256"]) == 64
            and (source_hash is None or receipt["source_sha256"] == source_hash)
            and receipt.get("task_ids") == ids
            and receipt.get("rows") == len(rows)
            and receipt.get("normalizations") == (expected_normalizations or [])
            and 1 <= len(rows) <= 50
            and len(ids) == len(set(ids))
        )
    except (OSError, ValueError, TypeError, KeyError):
        valid = False
    if not valid and external:
        raise ValueError("External prepared sample provenance is invalid or stale; it was not modified")
    return valid


def prepare_sample(
    destination: Path,
    *,
    source: Path | None = None,
    expected_source_sha256: str | None = None,
    identity_metadata_key: str = "task_id",
    normalize_image_detail: bool = False,
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
        if normalize_image_detail:
            from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

            for message in row["responses_create_params"]["input"]:
                if isinstance(message.get("content"), list):
                    for part in message["content"]:
                        if part.get("type") == "input_image":
                            part.setdefault("detail", "auto")
            NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
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
    receipt = {
        "source_sha256": source_hash,
        "output_sha256": output_hash,
        "task_ids": ids,
        "rows": len(rows),
        "normalizations": ["image_detail_auto"] if normalize_image_detail else [],
    }
    with tempfile.TemporaryDirectory(prefix="gym-frozen-sample-") as temporary:
        candidate = Path(temporary) / "sample.jsonl"
        candidate.write_bytes(output)
        stage_file(candidate, destination, expected_sha256=output_hash)
        proof = Path(temporary) / "provenance.json"
        proof.write_text(json.dumps(receipt, indent=2) + "\n")
        stage_file(proof, destination.with_suffix(".source.json"))
    return destination
