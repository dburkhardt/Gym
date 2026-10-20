# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and reuse the preselected sample; never change its identity or size."""
import json
import os
from pathlib import Path

def prepare() -> Path:
    path = Path(os.environ['CAMPAIGN_SAMPLE_JSONL']).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Stage pinned campaign sample first: {path}')
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not rows or len(rows)>50:
        raise ValueError('Expected1-50preselected sample rows')
    ids=[x['uuid'] for x in rows]
    if len(set(ids))!=len(ids):
        raise ValueError('Duplicate sample IDs')
    for row in rows:
        if not row.get('responses_create_params',{}).get('input') or 'verifier_metadata' not in row:
            raise ValueError('Missing solver input or isolated verifier metadata')
    return path

if __name__ == '__main__':
    print(prepare())
