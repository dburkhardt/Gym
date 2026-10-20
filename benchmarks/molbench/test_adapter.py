# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json,os
from pathlib import Path
from benchmarks.molbench.adapter import prepare_rows,score
REPO=Path(os.environ['MOLBENCH_SOURCE']); FROZEN=Path(os.environ['MOLBENCH_FROZEN'])
def test_frozen_source_prompts_and_reference_isolation():
 rows=prepare_rows(REPO,FROZEN);assert len(rows)==25
 assert [r['uuid'] for r in rows[:5]]==['10','17','24','14','13']
 assert all(set(r['responses_create_params'])=={'input'} for r in rows)
def test_native_positive_negative_duplicate_controls():
 row=prepare_rows(REPO,FROZEN)[0];gold=row['verifier_metadata']['sample']['answer']
 correct=score(REPO,row['verifier_metadata'],'<answer>'+json.dumps(gold[:3])+'</answer>')
 assert correct['reward']==1 and not correct['duplicate_ranking']
 empty=score(REPO,row['verifier_metadata'],'<answer>malformed</answer>')
 assert empty['reward']==0 and empty['parsed_count']==0
 repeated=score(REPO,row['verifier_metadata'],'<answer>'+json.dumps([gold[0]]*3)+'</answer>')
 assert repeated['reward']==1 and repeated['duplicate_ranking']
