# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One creator score per process so native SIGALRM scorers run on the main thread."""

import hashlib
import json
import sys
from pathlib import Path

from benchmarks.scientific_agent_campaign import deterministic


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text())
    evidence = None
    if request["benchmark"] == "qcbench":
        scorer_path = Path(request["repo"]) / "eval.py"
        evidence = {
            "creator_scorer_sha256": hashlib.sha256(scorer_path.read_bytes()).hexdigest(),
            "task_sha256": hashlib.sha256(
                json.dumps(request["task"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "source_revision": deterministic.PINS["qcbench"],
        }
    result = deterministic.score(request["benchmark"], Path(request["repo"]), request["task"], request["answer"])
    if evidence is not None:
        if hashlib.sha256(scorer_path.read_bytes()).hexdigest() != evidence["creator_scorer_sha256"]:
            raise RuntimeError("QCBench creator scorer changed during scoring")
        result["measurement_evidence"] = evidence
    source = Path(deterministic.__file__).resolve()
    Path(sys.argv[2]).write_text(
        json.dumps(
            {
                "details": result,
                "runtime": {
                    "source_path": str(source),
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                },
            }
        )
    )


if __name__ == "__main__":
    main()
