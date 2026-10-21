# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage all pinned public rows; output is planning data, not a runnable model route."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.wmdp_bio_chem.protocol import PROTOCOL, context
from nemo_gym.dataset_staging import stage_file


MANIFEST = Path(__file__).with_name("source-manifest.json")


def prepare(source: Path | None = None, destination: Path | None = None) -> Path:
    """Validate exact source bytes before publishing stable subset/index task identities."""
    import pyarrow.parquet as pq

    source = source or Path(os.environ["WMDP_DATA_ROOT"])
    destination = destination or Path(__file__).parent / "data"
    manifest = json.loads(MANIFEST.read_text())
    rows = []
    for subset, evidence in manifest["datasets"].items():
        path = source / f"{subset}.parquet"
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != evidence["sha256"]:
            raise ValueError(f"Pinned dataset unavailable or changed: {subset}")
        for index, document in enumerate(pq.read_table(path).to_pylist()):
            prompt = context(document, subset)
            task_id = f"{subset}:{index}"
            rows.append(
                {
                    "task_id": task_id,
                    "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
                    "verifier_metadata": {
                        "task_id": task_id,
                        "subset": subset,
                        "answer": document["answer"],
                        "protocol": PROTOCOL,
                        "dataset_revision": manifest["dataset_revision"],
                        "source_sha256": evidence["sha256"],
                        "context_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "continuations": [" A", " B", " C", " D"],
                    },
                }
            )
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temporary:
        staged = Path(temporary) / "benchmark.jsonl"
        staged.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        return stage_file(staged, destination / "benchmark.jsonl")


if __name__ == "__main__":
    prepare()
