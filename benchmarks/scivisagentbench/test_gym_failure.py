import asyncio,json,sys,tempfile,unittest,os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from responses_api_agents.scivis_native.app import Agent,RunRequest
class GymFailure(unittest.IsolatedAsyncioTestCase):
 async def test_judge_timeout_masked_and_logs_retained(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);(root/'tasks.json').write_text(json.dumps({'case1':{'source_case':{},'case_name':'case_1'}}));(root/'config.json').write_text('{}')
   script=root/'judge.py';script.write_text("import subprocess,sys,time\nprint('judge-start',flush=True)\nsubprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])\ntime.sleep(30)\n")
   fake=SimpleNamespace(_semaphore=asyncio.Semaphore(1),config=SimpleNamespace(inference_enabled=True,task_manifest=str(root/'tasks.json'),output_root=str(root/'out'),source_root='unused',native_config=str(root/'config.json'),endpoint='http://unused',max_tokens=16384,timeout_s=.15,judge_command=[sys.executable,str(script)]))
   async def native(source,config,task,output,*args):output.mkdir(parents=True);return {'success':True}
   body=RunRequest.model_validate({'responses_create_params':{'input':[]},'verifier_metadata':{'task_id':'case1'}})
   with patch('responses_api_agents.scivis_native.app.run_native',native), patch('responses_api_agents.scivis_native.app.os.killpg',wraps=os.killpg) as kill_group:
    result=await Agent.run(fake,body)
   kill_group.assert_called_once()
   self.assertTrue(result.mask_sample);self.assertIsNone(result.scientific_score);self.assertIn('TimeoutError',result.failure_reason)
   self.assertIn('judge-start',(Path(result.artifact_dir)/'native-judge.log').read_text())
   script.write_text("import sys\nprint('judge-error-receipt',flush=True)\nsys.exit(7)\n")
   fake.config.timeout_s=2
   with patch('responses_api_agents.scivis_native.app.run_native',native):failed=await Agent.run(fake,body)
   self.assertTrue(failed.mask_sample);self.assertIsNone(failed.scientific_score)
   self.assertIn('exited 7',failed.failure_reason)
   self.assertIn('judge-error-receipt',(Path(failed.artifact_dir)/'native-judge.log').read_text())
if __name__=='__main__':unittest.main()
