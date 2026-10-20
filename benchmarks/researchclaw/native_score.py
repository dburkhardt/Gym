# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrap creator score_workspace without replacing its metric or parser."""
import importlib, json, os, sys, threading
from pathlib import Path
MODEL='nvidia/nvidia/nemotron-3.5-super-vl'
def annotate(native, calls):
    failures=[i for i,c in enumerate(calls) if not isinstance(c.get('parsed'),dict) or 'score' not in c['parsed'] or isinstance(c['parsed'].get('score'),bool) or not isinstance(c['parsed'].get('score'),(float,int))]
    error=native.get('error')
    return {'native':native,'judge_calls':calls,'mask_sample':bool(failures or error),'failure_kind':'judge_failed' if failures else ('missing_report_or_native_error' if error else None),'invalid_judge_call_indices':failures,'reward':None if failures or error else native['total_score']/100,'qualification':'Native checklist agreement, not independent scientific correctness; raw creator metric preserved.'}
def score(source,workspace,output):
    if os.environ.get('JUDGE_MODEL_NAME') != MODEL: raise ValueError('Exact Super judge required before imports/calls')
    if not os.environ.get('JUDGE_API_BASE') or not os.environ.get('JUDGE_API_KEY'): raise ValueError('Explicit isolated judge transport required')
    sys.path.insert(0,str(Path(source).resolve()))
    module=importlib.import_module('evaluation.score'); original=module.LLMAgent; calls=[]; lock=threading.Lock()
    class Capturing:
        def __init__(self,*a,**kw):
            assert kw['model_version']==MODEL
            self.agent=original(*a,**kw)
        def __call__(self,*a,**kw):
            result=self.agent(*a,**kw)
            with lock:
                calls.append({'prompt':a[0] if a else None,'images':kw.get('image_paths'),'parsed':result})
                Path(output+'.calls.json').write_text(json.dumps(calls,indent=2))
            return result
    module.LLMAgent=Capturing
    try: native=module.score_workspace(workspace)
    except Exception as e: native={'error':type(e).__name__+': '+str(e)}
    finally: module.LLMAgent=original
    report=annotate(native,calls);Path(output).write_text(json.dumps(report,indent=2));return report
if __name__=='__main__': score(*sys.argv[1:])
