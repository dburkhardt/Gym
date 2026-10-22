# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest

from benchmarks.scivisagentbench.saved_judge import prepare_saved
from benchmarks.scivisagentbench.transport import MODEL


def fixture(tmp_path):
    root = tmp_path / 'original'
    task = {'case_name': 'case_1'}
    contents = {'identity.json': json.dumps({'task': task, 'model': MODEL}),
                'worker-input.json': '{}', 'solver/native-solver-result.json': '{"success":true}',
                'judge/case_1/test_results/result.json': '{}', 'workspace/screenshot.png': 'original pixels'}
    for rel, value in contents.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value)
    manifest = {'version': 1, 'purpose': 'saved-solver-judge-only', 'root': str(root),
                'files': {r: hashlib.sha256(v.encode()).hexdigest() for r, v in contents.items()}}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path, task, root


def test_saved_artifacts_preserved_exactly_without_solver(tmp_path):
    manifest, task, root = fixture(tmp_path)
    output = tmp_path / 'judging'
    _, solver = prepare_saved(manifest, task, output)
    assert solver['success']
    assert (output / 'exported-artifacts/screenshot.png').read_bytes() == (root / 'workspace/screenshot.png').read_bytes()
    assert not (output / 'workspace').exists()


@pytest.mark.parametrize('change', ['bytes', 'extra', 'task', 'symlink'])
def test_saved_judging_rejects_changed_or_incomplete_inputs(tmp_path, change):
    manifest, task, root = fixture(tmp_path)
    if change == 'bytes':
        (root / 'workspace/screenshot.png').write_text('changed')
    elif change == 'extra':
        (root / 'workspace/unlisted.png').write_text('unlisted')
    elif change == 'task':
        task['case_name'] = 'other'
    else:
        p = root / 'workspace/screenshot.png'
        p.unlink()
        p.symlink_to('/etc/passwd')
    with pytest.raises(ValueError):
        prepare_saved(manifest, task, tmp_path / 'judging')
