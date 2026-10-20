# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run native vision+text aggregate with synthetic images and mocked LLM only."""
import asyncio,json,sys
from pathlib import Path
from PIL import Image
from benchmarks.scivisagentbench.evaluate import evaluate
from benchmarks.scivisagentbench.transport import MODEL
async def main():
 source,out=map(Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=False)
 data=out/'data';data.mkdir();case=out/'case_1';case.mkdir()
 Image.new('RGB',(64,64),(0,128,0)).save(data/'gold.png');Image.new('RGB',(64,64),(0,128,0)).save(data/'result.png');(data/'answer.txt').write_text('Yes')
 assertions=[{'type':'llm-rubric','subtype':'vision','value':'1. Green object.\n2. Reference similarity.','gs-file':'gold.png','rs-file':'result.png'},{'type':'llm-rubric','subtype':'text','value':'1. Answer: Yes','rs-file':'answer.txt'}]
 calls=[]
 def sender(req):
  calls.append(req);vision=any(isinstance(m.get('content'),list) and any(c.get('type')=='image_url' for c in m['content']) for m in req['messages'])
  text={'goal_1_score':10,'goal_2_score':10,'total_score':20} if vision else {'score':10,'max_score':10}
  return {'model':MODEL,'choices':[{'finish_reason':'stop','message':{'content':json.dumps(text)}}]}
 result=await evaluate(source,{'assert':assertions},case,data,'mcp',sender,out/'evaluation')
 assert result['valid_for_interpretation'],result
 assert len(calls)==2
 (out/'control.json').write_text(json.dumps({'status':'passed','llm_calls':0,'mock_calls':2,'native_result':result,'scope':'Synthetic-image native aggregate control, not scientific judgement calibration'},indent=2))
asyncio.run(main())
