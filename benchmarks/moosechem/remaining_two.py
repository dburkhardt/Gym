# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen tasks0/33 only. Coordinator clearance required; no resume/retry."""
import fcntl,hashlib,json,os,subprocess,time,urllib.request,shutil
from pathlib import Path
from benchmarks.moosechem.native import claim_run,MODEL
from benchmarks.moosechem.admission import persist,gate
BASE=Path('/home/ubuntu/projects/broad-eval-campaign')

def tasks_from_sample(path):
 ids=[json.loads(s)['id'] for s in Path(path).read_text().splitlines()]
 assert ids==[18,26,38,0,33], 'Frozen calibration/expansion identity changed'
 return ids[-2:]


def validate_source(repo,pins):
 for name,sha in pins.items():assert hashlib.sha256((Path(repo)/name).read_bytes()).hexdigest()==sha, name

def validate_runtime(python,expected):
 probe="import sys,hashlib,importlib.metadata,json,pathlib; p=pathlib.Path(sys.executable).resolve(); print(json.dumps({'python_version':sys.version,'executable_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'packages':sorted((d.metadata['Name'].lower().replace('_','-'),d.version) for d in importlib.metadata.distributions())}))"
 actual=json.loads(subprocess.check_output([str(python),'-c',probe],text=True));assert actual==expected,'Runtime drift';return actual

def main():
 assert Path(__file__).resolve().parent==Path('/home/ubuntu/projects/overnight-gym-protein/benchmarks/moosechem'), 'Wrong executable module tree'
 root=BASE/'moosechem';out=root/'remaining2-admitted-v1';out.mkdir(exist_ok=True)
 lock=(root/'remaining2-admitted.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 sample=root/'sample5.jsonl';tasks=tasks_from_sample(sample);clear=json.loads((out/'slot-clearance.json').read_text())
 assert clear['approved'] is True and clear['producer']=='moosechem-remaining2' and 0<=time.time()-clear['issued_at']<=300
 assert clear['sample_sha256']==hashlib.sha256(sample.read_bytes()).hexdigest()
 for name,expected in clear['implementation_sha256'].items():
  assert name in ('native.py','admission.py','remaining_two.py','child_entry.py','source-sha256.json','runtime-fingerprint.json')
  assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==expected
 assert set(clear['implementation_sha256'])=={'native.py','admission.py','remaining_two.py','child_entry.py','source-sha256.json','runtime-fingerprint.json'}
 for task in tasks:
  for existing in root.glob(f'task{task}-*'):
   assert not any((existing/n).exists() for n in ('model-traces.jsonl','dispatch-intents.jsonl','run-intent.json')), 'Prior attempt requires reconciliation'
 validate_source(BASE/'moosechem-source',json.loads(Path(__file__).with_name('source-sha256.json').read_text()))
 validate_runtime(BASE/'moosechem-venv/bin/python',json.loads(Path(__file__).with_name('runtime-fingerprint.json').read_text()))
 claim_run(out)
 persist(out/'config.json',{'tasks':tasks,'model':MODEL,'all_stage_max_tokens':16384,'native_pipeline':'full150corpus; no screening/mutation/scorer reduction','variant':'16k all roles; per-request conservative admission; abort on first transport error'})
 for task in tasks:
  mem=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
  assert mem>=1024**3 and shutil.disk_usage(BASE).free>=8*1024**3
  ledger=(BASE/'inference-request-receipts.jsonl').read_bytes();assert not ledger or ledger.endswith(b'\n')
  assert gate([json.loads(s) for s in ledger.splitlines()],time.time(),16384+4096), 'Global drain/cooldown gate held; no new task'
  health=json.load(urllib.request.urlopen('http://127.0.0.1:49394/health',timeout=10));assert health['model']==MODEL and not health['held'] and health['code_sha256']==clear['implementation_sha256']['admission.py']
  attempt=root/f'task{task}-16k-admitted-v1';attempt.mkdir(exist_ok=False)
  command=[str(BASE/'moosechem-venv/bin/python'),'-u','-m','benchmarks.moosechem.native','--repo',str(BASE/'moosechem-source'),'--output',str(attempt),'--task-id',str(task),'--output-token-budget','16384']
  persist(attempt/'queue-dispatch-intent.json',{'time':time.time(),'task':task,'command':command})
  env=dict(os.environ,PYTHONPATH='/home/ubuntu/projects/overnight-gym-protein',MOOSE_KEY_FILE='/home/ubuntu/projects/skill-eval/.secrets/inference-nvidia.key',MOOSE_ADMISSION_URL='http://127.0.0.1:49394/v1')
  with (attempt/'native.log').open('x') as log:
   proc=subprocess.Popen([command[0],'-m','benchmarks.moosechem.child_entry',str(attempt),*command],env=env,cwd='/home/ubuntu/projects/overnight-gym-protein',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   deadline=time.time()+30
   while not (attempt/'child-identity.json').exists():
    assert proc.poll() is None and time.time()<deadline,'Child failed before durable identity'
    time.sleep(.05)
   ident=json.loads((attempt/'child-identity.json').read_text());assert ident['pid']==proc.pid and ident['pgid']==proc.pid
   persist(attempt/'parent-child-receipt.json',ident)
   persist(attempt/'child-release.json',{k:ident[k] for k in ('pid','start_ticks','boot_id')})
   code=proc.wait()
  persist(attempt/'queue-terminal.json',{'time':time.time(),'exit_code':code,'task':task,'result_exists':(attempt/'result.json').exists()})
  if code or not (attempt/'result.json').exists():
   persist(out/'terminal.json',{'status':'held_after_failure','task':task,'exit_code':code});return
 persist(out/'terminal.json',{'status':'completed','tasks':tasks})
if __name__=='__main__':main()
