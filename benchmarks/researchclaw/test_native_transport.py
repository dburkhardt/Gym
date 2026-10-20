# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real structai + OpenAI SDK, mocked HTTP only; never contacts a provider."""
import json,os,tempfile,unittest
from pathlib import Path
import httpx,openai,structai.llm_api
from native_score import score,MODEL
class NativeTransport(unittest.TestCase):
 def test_real_native_sdk(self):
  source=Path(os.environ['RESEARCHCLAW_SOURCE']).resolve()
  os.environ.update(JUDGE_MODEL_NAME=MODEL,JUDGE_API_BASE='http://mock.invalid/v1',JUDGE_API_KEY='mock')
  for case,masked in [('valid',False),('length',True),('wrong_model',True),('out_of_range',True)]:
   requests=[]
   def handler(request):
    requests.append(json.loads(request.content))
    return httpx.Response(200,json={'id':'mock-'+str(len(requests)),'object':'chat.completion','created':1,'model':MODEL if case!='wrong_model' else 'wrong','choices':[{'index':0,'finish_reason':'length' if case=='length' else 'stop','message':{'role':'assistant','content':json.dumps({'score':1000 if case=='out_of_range' else 50,'reasoning':'control'})}}],'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}})
   original=structai.llm_api.OpenAI
   structai.llm_api.OpenAI=lambda **kw:openai.OpenAI(api_key='mock',base_url='http://mock.invalid/v1',http_client=httpx.Client(transport=httpx.MockTransport(handler)))
   try:
    with tempfile.TemporaryDirectory() as d:
     w=Path(d)/'workspace';(w/'report').mkdir(parents=True)
     (w/'report/report.md').write_text('Synthetic transport control')
     (w/'capture-manifest.json').write_text('[]')
     (w/'_meta.json').write_text(json.dumps({'run_id':'control','task_id':'Life_002'}))
     out=str(Path(d)/'score.json');r=score(source,str(w),out)
     self.assertEqual(r['mask_sample'],masked,(case,r));self.assertTrue(all(q['model']==MODEL for q in requests))
     self.assertEqual(len(Path(out+'.provider.jsonl').read_text().splitlines()),len(requests))
     self.assertEqual(len(requests),10 if case in ('length','wrong_model') else 5)
   finally:structai.llm_api.OpenAI=original
if __name__=='__main__':unittest.main()
