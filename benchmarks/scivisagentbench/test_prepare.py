# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import pytest
from benchmarks.scivisagentbench import prepare as module


def test_changed_frozen_input_fails_before_publication(tmp_path):
    source = tmp_path / 'source.json'
    source.write_text('{}')
    with pytest.raises(ValueError, match='manifest changed'):
        module.prepare(source, tmp_path / 'published')
    assert not (tmp_path / 'published').exists()


def test_all_identities_preserved_without_judge_material_in_rows(tmp_path, monkeypatch):
    tasks = {f'napari-case{i}': {'case_name': f'case_{i}', 'source_case': {'assert': ['secret-rubric']}}
             for i in range(1, 12)}
    source = tmp_path / 'source.json'
    source.write_text(json.dumps(tasks))
    monkeypatch.setattr(module, 'FROZEN_SHA256', hashlib.sha256(source.read_bytes()).hexdigest())
    out = module.prepare(source, tmp_path / 'data')
    first = out.read_bytes()
    assert module.prepare(source, tmp_path / 'data').read_bytes() == first
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r['verifier_metadata']['task_id'] for r in rows] == list(tasks)
    assert len({r['task_id'] for r in rows}) == 11
    assert all(r['verifier_metadata']['frozen_manifest_sha256'] == module.FROZEN_SHA256 for r in rows)
    assert b'secret-rubric' not in first
    assert (out.parent / 'tasks.json').read_bytes() == source.read_bytes()
