# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio,json,sys
from pathlib import Path
from benchmarks.scivisagentbench.native import run_native
from benchmarks.scivisagentbench.transport import MODEL
async def main():
 source,packet,out=map(Path,sys.argv[1:]);task=json.loads((packet/'tasks.json').read_text())['napari-case1'];task['judge_case_dir']=str(out.parent/'mock-judge-case/case_1');config=json.loads((packet/'native-config.json').read_text());calls=[]
 async def sender(req):
  calls.append(req)
  return {'id':'mock-'+str(len(calls)),'model':MODEL,'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':'Mock transport control complete. No scientific answer or output was produced.'}}],'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
 result=await run_native(source,config,task,out,sender,16384)
 (out/'control.json').write_text(json.dumps({'llm_calls':0,'mock_calls':len(calls),'native_result':result,'scope':'Native agent setup/run/cleanup using real MCP channels and fake model responses; no scientific grading.'},indent=2))
 assert calls
asyncio.run(main())
