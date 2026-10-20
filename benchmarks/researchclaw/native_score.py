# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrap creator score_workspace without replacing its metric or parser."""
import importlib, json, os, sys, threading, math, tempfile
from pathlib import Path
MODEL='nvidia/nvidia/nemotron-3.5-super-vl'
def annotate(native, calls, expected_count=None, capture_omissions=None):
    failures=[i for i,c in enumerate(calls) if not isinstance(c.get('parsed'),dict) or 'score' not in c['parsed'] or isinstance(c['parsed'].get('score'),bool) or not isinstance(c['parsed'].get('score'),(float,int)) or not math.isfinite(c['parsed']['score']) or not 0 <= c['parsed']['score'] <= 100]
    error=native.get('error')
    expected_count = len(native.get('items', [])) if expected_count is None else expected_count
    coverage_error = expected_count <= 0 or len(calls) != expected_count or len(native.get('items', [])) != expected_count
    error = error or ('incomplete_checklist_coverage' if coverage_error else None) or ('capture_omissions' if capture_omissions else None)
    return {'native':native,'judge_calls':calls,'mask_sample':bool(failures or error),'failure_kind':'judge_failed' if failures else ('missing_report_or_native_error' if error else None),'invalid_judge_call_indices':failures,'reward':None if failures or error else native['total_score']/100,'qualification':'Native checklist agreement, not independent scientific correctness; raw creator metric preserved.'}
def score(source,workspace,output):
    if os.environ.get('JUDGE_MODEL_NAME') != MODEL: raise ValueError('Exact Super judge required before imports/calls')
    if not os.environ.get('JUDGE_API_BASE') or not os.environ.get('JUDGE_API_KEY'): raise ValueError('Explicit isolated judge transport required')
    # Creator writes _score.json here after judging: fail before any paid/API work.
    with tempfile.NamedTemporaryFile(prefix='.native-score-write-control-', dir=workspace):
        pass
    sys.path.insert(0,str(Path(source).resolve()))
    module=importlib.import_module('evaluation.score'); original=module.LLMAgent; calls=[]; lock=threading.Lock()
    class Capturing:
        def __init__(self,*a,**kw):
            assert kw['model_version']==MODEL
            kw['headers']={'X-Campaign-Benchmark':'ResearchClawBench'}
            self.agent=original(*a,**kw)
            # Disable SDK retries; creator's explicit max_try2 remains observable.
            if hasattr(self.agent, 'client'):
                self.agent.client.max_retries=0
                create=self.agent.client.chat.completions.create
                def guarded(**request):
                    if request.get('model') != MODEL: raise ValueError('Judge model request mismatch')
                    response=create(**request)
                    with lock:
                        with open(output+'.provider.jsonl','a') as receipt:
                            receipt.write(json.dumps({'request':request,'response':response.model_dump()})+'\n')
                    if response.model != MODEL: raise ValueError('Judge returned model mismatch')
                    if not response.choices or response.choices[0].finish_reason != 'stop' or not response.choices[0].message.content:
                        raise ValueError('Judge incomplete or empty response')
                    return response
                self.agent.client.chat.completions.create=guarded
        def __call__(self,*a,**kw):
            result=self.agent(*a,**kw)
            with lock:
                calls.append({'prompt':a[0] if a else None,'images':kw.get('image_paths'),'parsed':result})
                Path(output+'.calls.json').write_text(json.dumps(calls,indent=2))
            return result
    module.LLMAgent=Capturing
    original_item=module._score_single_item
    covered=[]
    def audit_item(agent,report,item,*args,**kwargs):
        with lock: covered.append(json.dumps(item,sort_keys=True))
        return original_item(agent,report,item,*args,**kwargs)
    module._score_single_item=audit_item
    try: native=module.score_workspace(workspace)
    except Exception as e: native={'error':type(e).__name__+': '+str(e)}
    finally:
        module.LLMAgent=original
        module._score_single_item=original_item
    meta=json.loads((Path(workspace)/'_meta.json').read_text())
    checklist=json.loads((Path(source)/'tasks'/meta['task_id']/'target_study/checklist.json').read_text())
    manifest=Path(workspace)/'capture-manifest.json'
    omissions=[x for x in json.loads(manifest.read_text()) if not x.get('copied')] if manifest.exists() else [{'reason':'capture manifest missing'}]
    if sorted(covered) != sorted(json.dumps(x,sort_keys=True) for x in checklist): omissions.append({'reason':'native criterion identity coverage mismatch'})
    report=annotate(native,calls,len(checklist),omissions); report['criterion_coverage']=covered;Path(output).write_text(json.dumps(report,indent=2));return report
if __name__=='__main__': score(*sys.argv[1:])
