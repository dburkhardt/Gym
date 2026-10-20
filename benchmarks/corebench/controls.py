# SPDX-License-Identifier: Apache-2.0
"""Prepare explicit oracle controls for the native Harbor runner; never dispatch a model."""

import json
import shlex
import shutil
from pathlib import Path

from benchmarks.corebench.prepare import TASK


def stage_control(native_tasks: Path, destination: Path, *, correct: bool) -> Path:
    """Keep creator task/scorer, add an oracle-only answer outside solver build context."""
    source = native_tasks / TASK
    if destination.exists():
        raise ValueError("Refuse to overwrite control task history")
    shutil.copytree(source, destination, symlinks=True)
    answers = json.loads((source / "tests/answers.json").read_text())
    report = dict(answers[0]) if correct else {key: "deliberately incorrect control" for key in answers[0]}
    solution = destination / "solution"
    solution.mkdir()
    payload = shlex.quote(json.dumps(report))
    (solution / "solve.sh").write_text(
        '#!/bin/bash\nset -eu\ntest ! -e /tests/answers.json\ntest ! -e /workspace/answers.json\ntest -z "$(find /workspace/capsule-9070543/results -type f -print -quit)"\nprintf %s '
        + payload
        + ' > /workspace/report.json\nprintf "reference isolation passed\\n" > /workspace/control-isolation.txt\n'
    )
    (solution / "solve.sh").chmod(0o755)
    return destination
