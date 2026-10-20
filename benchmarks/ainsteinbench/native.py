"""Creator DockerAgent with explicit answer-isolation and transport repairs."""
import argparse,copy,hashlib,importlib.util,json,os,pathlib,sys,time,uuid
from types import SimpleNamespace
MODEL='nvidia/nvidia/nemotron-3.5-super-vl'
IMAGES={2373:'shuoxin/pyscf_m_pyscf@sha256:c77a2bcdbc2446cefc9779ae57c84f20aa5e9f5741c16729c28d3231245e5d40',2396:'shuoxin/pyscf_m_pyscf@sha256:e65896a83ee4d9f350fcf0cefdad0c90109af1e8bfba9927f04dfa259deecb0d'}

def run(repo,out,task_id):
 repo,out=pathlib.Path(repo).resolve(),pathlib.Path(out).resolve()
 if out.exists():raise RuntimeError('Existing attempt requires explicit reconciliation')
 out.mkdir(parents=True)
 for relative,expected in json.loads(pathlib.Path(__file__).with_name('source-sha256.json').read_text()).items():
  assert hashlib.sha256((repo/relative).read_bytes()).hexdigest()==expected,relative
 sys.path.insert(0,str(repo))
 spec=importlib.util.spec_from_file_location('creator_ainstein_agent',repo/'scripts/run_agent.py');source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
 rows=[json.loads(x) for x in (repo/'data/questions/msb_converted.jsonl').read_text().splitlines()]
 raw=next(x for x in rows if x['question_id']==task_id);question=source.Question.from_dict(copy.deepcopy(raw))
 question.environment['working_directory']='/home/pyscf'
 image=IMAGES[int(raw['content']['pr_number'])]
 from openai import OpenAI
 client=OpenAI(api_key=pathlib.Path(os.environ['AINSTEIN_KEY_FILE']).read_text().strip(),base_url='http://127.0.0.1:49180/v1',max_retries=0,timeout=600,default_headers={'X-Campaign-Benchmark':'AInsteinBench'})
 def completion(**kwargs):
  assert kwargs['model']==MODEL
  assert 'max_tokens' not in kwargs
  kwargs['max_tokens']=16384
  record={'started':time.time(),'request':copy.deepcopy(kwargs)}
  try:
   response=client.chat.completions.create(**kwargs);assert response.model==MODEL
   record['response']=response.model_dump();return response
  except Exception as exc:
   record['error_type']=type(exc).__name__;raise
  finally:
   record['finished']=time.time()
   with (out/'model-traces.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 class Agent(source.DockerAgent):
  def initialize_container(self,question):
   self.container=self.docker_client.containers.run(image,command='sleep infinity',detach=True,working_dir='/home/pyscf',remove=False,network_disabled=True,mem_limit='4g',nano_cpus=1000000000,labels={'campaign':'ainsteinbench','task':task_id})
   head=self.container.exec_run(['git','rev-parse','HEAD']).output.decode().strip()
   assert head==raw['content']['base_commit'],(head,raw['content']['base_commit'])
   tracked=self.container.exec_run(['git','ls-files','-z']);assert tracked.exit_code==0
   files=tracked.output.decode().rstrip('\0').split('\0')
   hash_script="import hashlib,json,pathlib; files=json.loads(__import__('sys').argv[1]); print(json.dumps({f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest() for f in files},sort_keys=True))"
   before=self.container.exec_run(['python','-c',hash_script,json.dumps(files)]);assert before.exit_code==0,before.output
   # Keep base tree byte-identical; remove answer patches and reachable future git history.
   command='rm -f /home/fix.patch /home/test.patch /home/fix-run.sh /home/test-run.sh /home/run.sh; rm -rf .git; git init -q; git -c user.name=Benchmark -c user.email=benchmark@local add -A; git -c user.name=Benchmark -c user.email=benchmark@local commit -qm base; test ! -e /home/fix.patch; test ! -e /home/test.patch'
   result=self.container.exec_run(['bash','-lc',command]);assert result.exit_code==0,result.output
   after=self.container.exec_run(['python','-c',hash_script,json.dumps(files)]);assert after.exit_code==0,after.output
   assert before.output==after.output,'Base tracked-file content changed'
   history=self.container.exec_run(['git','rev-list','--count','HEAD']);assert history.output.strip()==b'1',history.output
   absent=self.container.exec_run(['bash','-lc','test ! -e /home/fix.patch && test ! -e /home/test.patch && test ! -e /home/fix-run.sh && test ! -e /home/test-run.sh && test ! -e /home/run.sh']);assert absent.exit_code==0
   (out/'base-file-hashes.json').write_bytes(before.output)
   (out/'container-control.json').write_text(json.dumps({'image':image,'base_commit':head,'answer_files_removed':True,'history_reinitialized':True,'history_commit_count':1,'tracked_file_count':len(files),'tracked_content_unchanged':True,'tracked_manifest_sha256':hashlib.sha256(before.output).hexdigest(),'network':'disabled','workdir':'/home/pyscf'},indent=2))
 agent=Agent(api_key='unused-placeholder',model=MODEL)
 agent.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
 (out/'protocol.json').write_text(json.dumps({'source_task':task_id,'source_task_sha256':hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest(),'variant':'corrected checkout path; evaluator files and future history hidden; network disabled; exact Super; explicit16k output budget','native_loop':'unchanged creator30iterations and4tools'},indent=2))
 try:
  answer=agent.solve_question(question);(out/(task_id+'.patch')).write_text(answer)
  (out/'solver-result.json').write_text(json.dumps({'status':'completed','patch_file':task_id+'.patch'}))
 except Exception as exc:
  (out/'failure.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc)}));raise
 finally:(out/'messages.json').write_text(json.dumps(agent.messages,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--task-id',required=True);a=p.parse_args();run(a.repo,a.output,a.task_id)
