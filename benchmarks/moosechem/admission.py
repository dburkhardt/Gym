# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact16k text-only MOOSE admission. No source/scorer changes or retries."""
import asyncio,json,os,time,uuid,hashlib
from pathlib import Path
import aiohttp
from aiohttp import web
CODE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
MODEL='nvidia/nvidia/nemotron-3.5-super-vl'
BASE=Path('/home/ubuntu/projects/broad-eval-campaign')
OUT=BASE/'moosechem/remaining2-admission'
UPSTREAM='http://127.0.0.1:49180'
serial=asyncio.Lock();held=False

def persist(path,value):
 global held
 try:
  with path.open('x') as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
  fd=os.open(path.parent,os.O_RDONLY)
  try:os.fsync(fd)
  finally:os.close(fd)
 except (OSError,ValueError,TypeError):held=True;raise

def reservation(body):
 budgets=[body[k] for k in ('max_tokens','max_completion_tokens','max_output_tokens') if k in body]
 if not budgets or any(type(v)is not int or v!=16384 for v in budgets):raise ValueError('Exact16384 budget required')
 if not isinstance(body.get('messages'),list) or any(not isinstance(m.get('content'),str) for m in body['messages']):raise ValueError('Only native text messages supported')
 value=16384+4096+len(json.dumps(body,ensure_ascii=False).encode())
 if value>100000:raise ValueError('request_exceeds_admission_capacity')
 return value

def gate(receipts,now,reserve):
 done=[r for r in receipts if r.get('event')=='finished'];finished={r.get('request_id') for r in done}
 if any(r.get('event')=='started' and r.get('request_id') not in finished for r in receipts):return False
 if any(r.get('http_status')==429 and now-r['time']<max(120,r.get('retry_after_seconds') or 0) for r in done):return False
 recent=[r for r in done if 0<=now-r['time']<60]
 if any(r.get('http_status')==200 and type((r.get('usage') or {}).get('total_tokens'))is not int for r in recent):return False
 return sum((r.get('usage') or {}).get('total_tokens',0) or 0 for r in recent)+reserve<=100000

async def handle(request):
 global held
 if request.path=='/health':return web.json_response({'held':held or (OUT/'held.json').exists(),'model':MODEL,'code_sha256':CODE_SHA256})
 if request.method!='POST' or request.path!='/v1/chat/completions':raise web.HTTPNotFound()
 body=await request.json()
 if body.get('model')!=MODEL or body.get('stream') or request.headers.get('X-Campaign-Benchmark')!='MOOSE-Chem':raise web.HTTPForbidden()
 try:reserve=reservation(body)
 except (ValueError,TypeError,KeyError) as e:raise web.HTTPBadRequest(text=str(e))
 async with serial:
  if held or (OUT/'held.json').exists():raise web.HTTPServiceUnavailable(text='Review required; no dispatch')
  artifact=OUT/uuid.uuid4().hex;artifact.mkdir();persist(artifact/'request.json',body);deadline=time.time()+300
  while True:
   try:
    raw=(BASE/'inference-request-receipts.jsonl').read_bytes();assert not raw or raw.endswith(b'\n');allowed=gate([json.loads(x) for x in raw.splitlines()],time.time(),reserve)
   except (OSError,ValueError,TypeError,AssertionError,KeyError):allowed=False
   if allowed:break
   if time.time()>deadline:
    persist(artifact/'terminal.json',{'status':'admission_timeout','dispatches':0});raise web.HTTPServiceUnavailable(text='Admission timed out; no dispatch')
   await asyncio.sleep(5)
  persist(artifact/'started.json',{'time':time.time(),'status':'dispatch-intent','reservation':reserve})
  try:
   async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as client:
    async with client.post(UPSTREAM+request.path,json=body,headers={'Authorization':request.headers.get('Authorization',''),'X-Campaign-Benchmark':'MOOSE-Chem'}) as response:
     payload=await response.read()
     try:data=json.loads(payload)
     except ValueError:data={'raw_response_text':payload.decode(errors='replace')}
     if not isinstance(data,dict):data={'unexpected_json_response':data}
     persist(artifact/'response.json',data)
     usage=(data.get('usage') or {}).get('total_tokens')
     if response.status!=200 or data.get('model')!=MODEL or type(usage)is not int or usage>reserve or usage<0:
      held=True;persist(OUT/'held.json',{'status':response.status,'actual_usage':usage,'reservation':reserve,'requires_reconciliation':True})
     persist(artifact/'terminal.json',{'status':response.status,'time':time.time(),'dispatches':1})
     return web.Response(body=payload,status=response.status,content_type='application/json',headers={'Retry-After':response.headers['Retry-After']} if 'Retry-After' in response.headers else {})
  except (aiohttp.ClientError,asyncio.TimeoutError) as exc:
   held=True;persist(OUT/'held.json',{'error':type(exc).__name__,'outcome':'unknown','requires_reconciliation':True});persist(artifact/'terminal.json',{'status':'transport_error','outcome':'unknown','dispatches':1});raise web.HTTPBadGateway(text='Transport failed; no retry')

if __name__=='__main__':
 import fcntl
 OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'process.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 # Refuse uncertain recovery after process termination, even without a held latch.
 assert not any(p.is_dir() and not (p/'terminal.json').exists() for p in OUT.iterdir()),'Unresolved request requires reconciliation'
 app=web.Application(client_max_size=16*1024**2);app.router.add_route('*','/{path:.*}',handle);web.run_app(app,host='127.0.0.1',port=49394,access_log=None)
