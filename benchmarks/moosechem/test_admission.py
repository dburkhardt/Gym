# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio,json,tempfile,unittest
from pathlib import Path
from aiohttp import web,ClientSession
from aiohttp.test_utils import TestServer
from benchmarks.moosechem import admission as p
class ProxyHTTP(unittest.IsolatedAsyncioTestCase):
 async def asyncSetUp(self):
  self.tmp=tempfile.TemporaryDirectory();p.BASE=Path(self.tmp.name)/'campaign';p.BASE.mkdir();p.OUT=p.BASE/'admission';p.OUT.mkdir();p.serial=asyncio.Lock();p.held=False
  (p.BASE/'inference-request-receipts.jsonl').write_text(''); key=p.BASE.parent/'skill-eval/.secrets';key.mkdir(parents=True);(key/'inference-nvidia.key').write_text('test-only')
  self.calls=0;self.status=429;self.drop=False;self.payload={'error':'mock'}
  async def upstream(req):
   self.calls+=1
   if self.drop:req.transport.close();return web.Response()
   return web.json_response(self.payload,status=self.status,headers={'Retry-After':'17'})
  app=web.Application();app.router.add_post('/v1/chat/completions',upstream);self.up=TestServer(app);await self.up.start_server();p.UPSTREAM=str(self.up.make_url('')).rstrip('/')
  app=web.Application();app.router.add_route('*','/{path:.*}',p.handle);self.proxy=TestServer(app);await self.proxy.start_server();self.client=ClientSession()
 async def asyncTearDown(self):
  await self.client.close();await self.proxy.close();await self.up.close();self.tmp.cleanup()
 async def send(self,**overrides):
  body={'model':p.MODEL,'max_tokens':16384,'messages':[]};body.update(overrides)
  async with self.client.post(self.proxy.make_url('/v1/chat/completions'),json=body,headers={'X-Campaign-Benchmark':'MOOSE-Chem','Authorization':'secret-not-recorded'}) as r:return r.status,await r.text(),dict(r.headers)
 async def test_rejections_no_dispatch(self):
  for kw in ({'model':'wrong'},{'stream':True},{'max_completion_tokens':8192},{'max_tokens':True}):self.assertIn((await self.send(**kw))[0],(400,403))
  self.assertEqual(self.calls,0)
 async def test_429_once_and_header(self):
  status,_,headers=await self.send();self.assertEqual(status,429);self.assertEqual(headers['Retry-After'],'17');self.assertEqual(self.calls,1)
  self.assertNotIn('secret-not-recorded',''.join(f.read_text() for f in p.OUT.rglob('*') if f.is_file()))
 async def test_transport_once(self):
  self.drop=True;status,_,_=await self.send();self.assertEqual(status,502);self.assertEqual(self.calls,1)
  self.assertEqual((await self.send())[0],503);self.assertEqual(self.calls,1)
 async def test_retry_is_latched_no_second_dispatch(self):
  self.assertEqual((await self.send())[0],429)
  self.assertEqual((await self.send())[0],503)
  self.assertEqual(self.calls,1)
 async def test_queued_request_rechecks_latch(self):
  a,b=await asyncio.gather(self.send(),self.send())
  self.assertEqual(sorted([a[0],b[0]]),[429,503]);self.assertEqual(self.calls,1)
 async def test_usage_overrun_holds_queued(self):
  self.status=200;self.payload={'model':p.MODEL,'usage':{'total_tokens':200000}}
  a,b=await asyncio.gather(self.send(),self.send());self.assertEqual(sorted([a[0],b[0]]),[200,503]);self.assertEqual(self.calls,1)
 async def test_missing_usage_holds_next(self):
  self.status=200;self.payload={'model':p.MODEL}
  self.assertEqual((await self.send())[0],200);self.assertEqual((await self.send())[0],503);self.assertEqual(self.calls,1)
 async def test_oversized_request_fails_without_waiting(self):
  self.assertEqual((await self.send(messages=[{'role':'user','content':'x'*100000}]))[0],400)
  self.assertEqual(self.calls,0)
 async def test_exact_success_and_durable_intent(self):
  self.status=200;self.payload={'model':p.MODEL,'usage':{'total_tokens':100}}
  self.assertEqual((await self.send())[0],200)
  artifacts=[x for x in p.OUT.iterdir() if x.is_dir()];self.assertEqual(len(artifacts),1)
  self.assertEqual(json.loads((artifacts[0]/'started.json').read_text())['status'],'dispatch-intent')
  self.assertEqual(json.loads((artifacts[0]/'terminal.json').read_text())['dispatches'],1)
if __name__=='__main__':unittest.main()
