# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare deterministic campaign subsets from already downloaded, pinned sources."""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def stratified(rows: list[dict], key: str, count: int, seed: int = 20261020) -> list[dict]:
    """Round-robin shuffled strata without looking at model performance."""
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, 'unknown'))].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < min(count, len(rows)):
        for name in sorted(groups):
            if groups[name] and len(selected) < count:
                selected.append(groups[name].pop())
    return selected


def prepare_scicode(source: Path, output: Path) -> dict:
    """Preserve native Gym fields, including complete substeps and dependencies."""
    raw = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(raw) == 65, f'Unexpected pinned test size: {len(raw)}'
    ids = [str(x['problem_id']) for x in raw]
    assert len(set(ids)) == len(ids)
    rows = []
    for entry in raw:
        for step in entry['sub_steps']:
            assert step.get('step_description_prompt') and isinstance(step.get('test_cases'), list)
        rows.append({
            'responses_create_params': {'input': []},
            'problem_id': entry['problem_id'], 'problem_name': entry['problem_name'],
            'required_dependencies': entry['required_dependencies'],
            'sub_steps': entry['sub_steps'], 'uuid': entry['problem_id'],
            '_campaign_stratum': str(len(entry['sub_steps'])),
        })
    selected = stratified(rows, '_campaign_stratum', 50)
    for row in selected:
        row.pop('_campaign_stratum')
    output.mkdir(parents=True, exist_ok=True)
    for n in (5, 20, 50):
        (output / f'scicode-{n}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in selected[:n]))
    return {'benchmark': 'SciCode', 'revision': '4510f6a6aa27c43fad7b43da2c59602a86e88480',
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'empty_test_substeps': [s['step_number'] for x in raw for s in x['sub_steps'] if not s['test_cases']],
            'population': 65, 'substeps': sum(len(x['sub_steps']) for x in raw),
            'selected_ids': [x['uuid'] for x in selected], 'stratification': 'substep count',
            'held_out_ids': [x['uuid'] for x in selected[20:]],
            'protocol': 'Existing Gym background prompt; not an unchanged creator prompt',
            'required_private_grader_artifact': 'test_data.h5',
            'artifact_sha256': '48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_scicode(args.source, args.output)
    (args.output / 'preparation.json').write_text(json.dumps(manifest, indent=2))
