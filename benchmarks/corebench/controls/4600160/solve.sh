#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eu
test ! -e /tests/answers.json
test ! -e /workspace/answers.json
test ! -S /var/run/docker.sock
test -z "$(find -L /results -mindepth 1 -maxdepth 1 -print -quit)"
cat /proc/mounts > /workspace/scientific-mounts.txt
cat /proc/net/route > /workspace/scientific-routes.txt
cd /workspace/capsule-4600160
sha256sum -c /solution/source.sha256 > /workspace/scientific-source-before.log
cd /code
# Execute the complete creator gaze and pupil walkthrough on its original inputs.
bash run.sh > /workspace/scientific-workflow.log 2>&1
cd /workspace/capsule-4600160
sha256sum -c /solution/source.sha256 > /workspace/scientific-source-after.log
/opt/corebench-harness/bin/python3 - <<'PY'
import json
import struct
from pathlib import Path

dimensions = {}
for name in ("gaze.png", "pupil.png"):
    data = (Path("/results") / name).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", name
    assert data[12:16] == b"IHDR", name
    width, height = struct.unpack(">II", data[16:24])
    assert width > 0 and height > 0, name
    dimensions[name] = {"width": width, "height": height, "bytes": len(data)}
Path("/workspace/scientific-figures.json").write_text(json.dumps(dimensions, indent=2) + "\n")
PY
# Figure existence and original execution are operational evidence, not model scores.
printf '{}\n' > /workspace/report.json
