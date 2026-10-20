# SPDX-License-Identifier: Apache-2.0
"""Prepare explicit oracle controls for the native Harbor runner; never dispatch a model."""

import json
import shlex
import shutil
import statistics
from pathlib import Path

from benchmarks.corebench.prepare import TASK


def stage_control(native_tasks: Path, destination: Path, *, correct: bool, task_id: str = TASK) -> Path:
    """Keep creator task/scorer, add an oracle-only answer outside solver build context."""
    if Path(task_id).name != task_id:
        raise ValueError("Invalid capsule identifier")
    source = native_tasks / task_id
    if destination.exists():
        raise ValueError("Refuse to overwrite control task history")
    shutil.copytree(source, destination, symlinks=True)
    answers = json.loads((source / "tests/answers.json").read_text())
    report = (
        {
            key: statistics.mean(values) if all(type(v) in (int, float) for v in values) else values[0]
            for key in answers[0]
            for values in [[answer[key] for answer in answers]]
        }
        if correct
        else {key: "deliberately incorrect control" for key in answers[0]}
    )
    solution = destination / "solution"
    solution.mkdir()
    payload = shlex.quote(json.dumps(report))
    (solution / "solve.sh").write_text(
        f'#!/bin/bash\nset -eu\ntest ! -e /tests/answers.json\ntest ! -e /workspace/answers.json\ntest -z "$(find /workspace/{task_id}/results -type f -print -quit)"\nprintf %s '
        + payload
        + ' > /workspace/report.json\nprintf "reference isolation passed\\n" > /workspace/control-isolation.txt\nif [ -d /opt/corebench-build-metadata ]; then cp -r /opt/corebench-build-metadata /workspace/corebench-build-metadata; fi\n'
    )
    (solution / "solve.sh").chmod(0o755)
    return destination
