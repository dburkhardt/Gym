# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report original creator accuracy/RMSE, never a mean of mixed transport rewards."""
import argparse
import json
from pathlib import Path
from benchmarks.scientific_agent_campaign.deterministic import module, validate_repo

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--rollouts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    validate_repo('fgbench',a.repo)
    rows=[json.loads(x) for x in a.rollouts.read_text().splitlines()]
    answers=[''.join(c['text'] for m in r['response']['output'] if m['type']=='message' for c in m['content'] if c['type']=='output_text') for r in rows]
    scorer=module(a.repo/'result_ananlysis/analysis.py')
    metadata=[r['verifier_metadata'] for r in rows]
    if set(scorer.task_name(r) for r in metadata)!=set(scorer.TASK_COUNTS):
        raise ValueError('Pilot does not cover all6creator task strata; report partial coverage, not full aggregate.')
    a.output.write_text(json.dumps(scorer.evaluate(metadata,answers,'standard'),indent=2))
