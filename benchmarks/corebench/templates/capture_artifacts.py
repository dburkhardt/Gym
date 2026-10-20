# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy bounded scientific outputs to verifier logs before Harbor removes a sandbox."""

import hashlib
import json
import shutil
from pathlib import Path


root = Path("/workspace")
output = Path("/logs/verifier/workspace-artifacts")
output.mkdir(parents=True, exist_ok=True)
budget = 250 * 1024 * 1024
records = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.is_symlink():
        continue
    relative = path.relative_to(root)
    if relative.parts[0] == "assets":
        continue  # immutable input assets are already preserved with the source task
    size = path.stat().st_size
    entry = {"path": str(relative), "bytes": size, "copied": False}
    if size <= min(100 * 1024 * 1024, budget):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        entry["sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        entry["copied"] = True
        budget -= size
    else:
        entry["omitted_reason"] = "artifact size budget"
    records.append(entry)
(output / "capture-manifest.json").write_text(json.dumps(records, indent=2))
