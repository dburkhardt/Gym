# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise actual creator score_workspace with a no-network structai test double."""
import json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from native_score import score,MODEL
SOURCE=Path(os.environ['RESEARCHCLAW_SOURCE']).resolve()
class Fake:
    response={'score':50,'reasoning':'control'}
    settings=[]
    def __init__(self,**kw):self.settings.append(kw)
    def __call__(self,*a,**kw):return self.response
class Native(unittest.TestCase):
    def setUp(self):
        self.env_patch=patch.dict(os.environ,{})
        self.env_patch.start()
        self.old_evaluation={k:v for k,v in sys.modules.items() if k=='evaluation' or k.startswith('evaluation.')}
        for k in self.old_evaluation:sys.modules.pop(k,None)
        self.module_patch=patch.dict(sys.modules,{'structai':types.SimpleNamespace(LLMAgent=Fake,multi_thread=lambda inputs,func,**kw:[func(**i) for i in inputs])})
        self.module_patch.start()
        Fake.settings=[]
    def tearDown(self):
        self.module_patch.stop()
        for k in list(sys.modules):
            if k=='evaluation' or k.startswith('evaluation.'):sys.modules.pop(k,None)
        sys.modules.update(self.old_evaluation)
        self.env_patch.stop()

    def test_actual_native_valid_and_parse_failure(self):
        os.environ.update(JUDGE_MODEL_NAME=MODEL,JUDGE_API_BASE='http://not-called.invalid',JUDGE_API_KEY='control')
        for value,masked in [({'score':50,'reasoning':'control'},False),(None,True),({},True)]:
            with tempfile.TemporaryDirectory() as d:
                w=Path(d)/'workspace';w.mkdir();(w/'report').mkdir();(w/'report/report.md').write_text('Synthetic control report, not model output')
                (w/'capture-manifest.json').write_text('[]')
                (w/'_meta.json').write_text(json.dumps({'run_id':'mock','task_id':'Life_002'}))
                Fake.response=value
                r=score(SOURCE,str(w),str(Path(d)/'receipt.json'))
                self.assertEqual(r['mask_sample'],masked);self.assertEqual(len(r['native']['items']),5)
                self.assertEqual(r['native']['total_score'],0 if masked else 50)
                self.assertEqual(len(r['judge_calls']),5)
        self.assertTrue(all(x['model_version']==MODEL and x['max_tokens']==500 for x in Fake.settings))
    def test_wrong_model_rejected_before_native_call(self):
        os.environ['JUDGE_MODEL_NAME']='wrong'
        with self.assertRaises(ValueError):score(SOURCE,'unused','unused')
if __name__=='__main__':unittest.main()
