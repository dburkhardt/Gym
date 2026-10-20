# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze public inputs and exact creator instructions; never execute an agent."""
import argparse, hashlib, json, runpy, shutil
from pathlib import Path
TASKS = ('Neuroscience_000', 'Life_002')
MODEL = 'nvidia/nvidia/nemotron-3.5-super-vl'
def hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file()}
def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    template = runpy.run_path(str(source/'evaluation/instructions_tmpl.py'))['INSTRUCTIONS_TEMPLATE']
    manifest = {'tasks':{}, 'model':MODEL, 'scope':'Two calibration tasks, not full40-task benchmark', 'remaining_task_ids':sorted(p.name for p in (source/'tasks').iterdir() if p.is_dir() and p.name not in TASKS)}
    for name in TASKS:
        src=source/'tasks'/name; task=output/name; env=task/'environment'; ws=env/'workspace'
        ws.mkdir(parents=True)
        info=json.loads((src/'task_info.json').read_text())
        for folder in ('data','related_work'):
            if (src/folder).exists(): shutil.copytree(src/folder,ws/folder)
        for folder in ('code','outputs','report/images'): (ws/folder).mkdir(parents=True,exist_ok=True)
        parts=[]
        for d in info.get('data',[]):
            typ=f" [{d['type']}]" if d.get('type') else ''
            parts.append(f"- **{d['name']}**{typ} (`{d.get('path','').lstrip('./')}`): {d.get('description','')}")
        prompt=template.format(workspace='/workspace',task_desc=info.get('task',''),data_text='\n'.join(parts) or 'No specific data files.')
        (task/'instruction.md').write_text(prompt); (ws/'INSTRUCTIONS.md').write_text(prompt)
        (ws/'_meta.json').write_text(json.dumps({'run_id':f'calibration-{name}','task_id':name,'agent_name':MODEL}))
        (env/'Dockerfile').write_text('FROM researchclaw-tools:campaign\nCOPY workspace /workspace\nRUN chmod -R a-w /workspace/data /workspace/related_work\nWORKDIR /workspace\n')
        (task/'task.toml').write_text('version = "1.0"\n[agent]\ntimeout_sec = 7200\n[verifier]\ntimeout_sec = 600\n[environment]\ncpus = 2\nmemory_mb = 4096\nstorage_mb = 4096\nallow_internet = false\n')
        tests=task/'tests'; tests.mkdir()
        shutil.copy2(Path(__file__).with_name('capture.py'),tests/'capture.py')
        (tests/'test.sh').write_text('#!/bin/bash\nset -eu\npython /tests/capture.py\n')
        # References remain outside solver build context. Native scoring is a host sidecar.
        ref=output/'judge-only'/name; ref.mkdir(parents=True)
        shutil.copytree(src/'target_study',ref/'target_study')
        manifest['tasks'][name]={'public_inputs':hashes(ws),'references':hashes(ref),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'checklist_items':len(json.loads((ref/'target_study/checklist.json').read_text()))}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args();prepare(a.source,a.output)
