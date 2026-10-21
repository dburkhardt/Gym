# SPDX-License-Identifier: Apache-2.0
"""Publish frozen task identities without exposing judge references in model inputs."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_gym.dataset_staging import stage_file

FROZEN_SHA256 = '224e99d47eeb90adc106f1b11e0fa2309d28cae5a4e44144e14a99006449e689'


def prepare(source: Path | None = None, destination: Path | None = None) -> Path:
    source = source or Path(os.environ['SCIVIS_FROZEN_TASKS_JSON'])
    destination = destination or Path(__file__).parent / 'data'
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != FROZEN_SHA256:
        raise ValueError('Frozen SciVis manifest changed')
    tasks = json.loads(content)
    expected = [f'napari-case{i}' for i in range(1, 12)]
    if set(tasks) != set(expected):
        raise ValueError('All eleven frozen SciVis cases are required')
    rows = []
    for key in expected:
        task = tasks[key]
        rows.append({
            'task_id': f'scivisagentbench::{key}',
            'responses_create_params': {'input': []},
            'verifier_metadata': {'task_id': key, 'frozen_manifest_sha256': FROZEN_SHA256},
            'agent_ref': {'name': 'scivis_agent'},
        })
        if task['case_name'] != 'case_' + key.removeprefix('napari-case'):
            raise ValueError('Frozen case identity mismatch')
    # The host-only manifest retains rubrics and historical runtime paths.
    # It is not a solver upload or evidence that those paths exist on this host.
    stage_file(source, destination / 'tasks.json', expected_sha256=FROZEN_SHA256)
    with TemporaryDirectory() as tmp:
        candidate = Path(tmp) / 'benchmark.jsonl'
        candidate.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return stage_file(candidate, destination / 'benchmark.jsonl')


if __name__ == '__main__':
    print(prepare(Path(os.environ['SCIVIS_FROZEN_TASKS_JSON']), Path(__file__).parent / 'data'))
