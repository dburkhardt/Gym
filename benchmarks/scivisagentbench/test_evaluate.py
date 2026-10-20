import asyncio,json,tempfile,unittest
from pathlib import Path
from benchmarks.scivisagentbench.evaluate import evaluate, aggregate_validity
import copy
from benchmarks.scivisagentbench.transport import MODEL
SOURCE=Path(__file__).resolve().parents[4]/'work/scivisagentbench-upstream'
class NativeAggregate(unittest.IsolatedAsyncioTestCase):
 async def test_actual_text_aggregate_and_failure_coverage(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);case=root/'case_1';case.mkdir();(root/'answer.txt').write_text('Yes')
   source_case={'assert':[{'type':'llm-rubric','subtype':'text','value':'1. Q1 correct answer: Yes','rs-file':'answer.txt'}]}
   calls=[]
   def sender(req):
    calls.append(req);return {'model':MODEL,'choices':[{'finish_reason':'stop','message':{'content':'{"score":10,"max_score":10}'}}]}
   r=await evaluate(SOURCE,source_case,case,root,'mcp',sender,root/'good');self.assertTrue(r['valid_for_interpretation']);self.assertEqual(len(calls),1);self.assertEqual(calls[0]['max_tokens'],2000)
   dropped=copy.deepcopy(r['native_result']);dropped['scores']['total_score']=0
   self.assertFalse(aggregate_validity(dropped,r['parsed_judgements'],source_case['assert'])['valid'])
   (root/'answer.txt').unlink();r=await evaluate(SOURCE,source_case,case,root,'mcp',sender,root/'missing');self.assertFalse(r['valid_for_interpretation']);self.assertEqual(len(calls),1)
   (root/'answer.txt').write_text('Yes')
   def bad(req):return {'model':MODEL,'choices':[{'finish_reason':'stop','message':{'content':'{"score":1000,"max_score":10}'}}]}
   r=await evaluate(SOURCE,source_case,case,root,'mcp',bad,root/'invalid');self.assertFalse(r['valid_for_interpretation']);self.assertEqual(r['parsed_judgements'][0]['native_result']['score'],1000)
if __name__=='__main__':unittest.main()
