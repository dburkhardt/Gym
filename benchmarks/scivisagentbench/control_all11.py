"""Exact native case assertions with synthetic output files/fake model, never scientific results."""
import asyncio,json,sys
from pathlib import Path
from PIL import Image
from benchmarks.scivisagentbench.evaluate import evaluate
from benchmarks.scivisagentbench.transport import MODEL
async def main():
 source,manifest,out=map(Path,sys.argv[1:]);tasks=json.loads(manifest.read_text());out.mkdir(exist_ok=False);reports=[]
 for name,task in tasks.items():
  case=task['source_case'];base=out/name;data=base/'data';casedir=base/task['case_name'];casedir.mkdir(parents=True);data.mkdir();mode=task['agent_mode']
  for a in case['assert']:
   result=data/a['rs-file'].replace('{agent_mode}',mode);result.parent.mkdir(parents=True,exist_ok=True)
   if a['subtype']=='text':result.write_text('Yes' if name!='napari-case7' else '2')
   else:
    Image.new('RGB',(64,64),(0,128,0)).save(result)
    if a.get('gs-file'):
     ref=data/a['gs-file'];ref.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(64,64),(0,128,0)).save(ref)
  calls=[]
  def sender(req):
   a=case['assert'][len(calls)];calls.append(req);n=len([x for x in a['value'].splitlines() if x.strip()]);answer={'score':10*n,'max_score':10*n}
   if a['subtype']=='vision':answer={**{f'goal_{i}_score':10 for i in range(1,n+1)},'total_score':10*n}
   return {'id':f'mock-{name}-{len(calls)}','model':MODEL,'choices':[{'finish_reason':'stop','message':{'content':json.dumps(answer)}}]}
  result=await evaluate(source,case,casedir,data,mode,sender,base/'evaluation')
  assert result['valid_for_interpretation'],(name,result)
  reports.append({'case':name,'mock_calls':len(calls),'expected':len(case['assert']),'valid':True,'native_score':result['native_result']['scores'],'llm_calls':0})
 (out/'controls.json').write_text(json.dumps({'scope':'Synthetic files and fake judge; exact source assertions and native aggregate. Not scientific validation.','cases':reports},indent=2))
asyncio.run(main())
