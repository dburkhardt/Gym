# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harbor verifier phase captures outputs; it never invents a scientific score."""
import hashlib,json,shutil
from pathlib import Path
def capture(root,output,budget=250*1024*1024):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True);records=[]
    for path in sorted(root.rglob('*')):
        rel=path.relative_to(root)
        if rel.parts[0] in ('data','related_work'):continue
        if path.is_symlink():
            records.append({'path':str(rel),'copied':False,'reason':'symlink excluded'});continue
        if not path.is_file():continue
        size=path.stat().st_size;record={'path':str(rel),'bytes':size,'copied':False}
        if size<=min(100*1024*1024,budget):
            dest=output/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
            record.update(copied=True,sha256=hashlib.sha256(dest.read_bytes()).hexdigest());budget-=size
        else:record['reason']='artifact size budget'
        records.append(record)
    (output/'capture-manifest.json').write_text(json.dumps(records,indent=2))
    return records
if __name__=='__main__':
    out=Path('/logs/verifier');records=capture('/workspace',out/'workspace-artifacts')
    missing=[x for x in records if not x['copied']]
    (out/'grading-status.json').write_text(json.dumps({'status':'pending_host_native_judge','scientific_score':None,'capture_omissions':missing}))
    # Harbor requires a verifier result; this named operational metric is NOT accuracy.
    (out/'reward.json').write_text(json.dumps({'artifact_capture_complete':int(not missing)}))
