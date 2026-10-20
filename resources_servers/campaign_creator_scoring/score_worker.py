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
    result = deterministic.score(request["benchmark"], Path(request["repo"]), request["task"], request["answer"])
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
