import asyncio,json,os
from pathlib import Path
from typing import ClassVar
from nemo_gym.base_resources_server import BaseResourcesServerConfig,BaseVerifyRequest,BaseVerifyResponse,ReverifyMode,SimpleResourcesServer
class Config(BaseResourcesServerConfig):
    REVERIFY_MODE:ClassVar[ReverifyMode]=ReverifyMode.STATELESS
class Request(BaseVerifyRequest):
    id:str
    verifier_metadata:dict
class Response(BaseVerifyResponse,Request):
    metrics:dict|None
    scoring_note:str
    scoring_diagnostics:dict
class Server(SimpleResourcesServer):
    config:Config
    ray_enabled=False
    async def verify(self,body:Request)->Response:
        payload={**body.verifier_metadata,'answer':body.response.output_text or ''}
        p=await asyncio.create_subprocess_exec(os.environ['BIOPLANNER_SCORER_PYTHON'],str(Path(__file__).with_name('score_worker.py')),stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        try:
            stdout,stderr=await asyncio.wait_for(p.communicate(json.dumps(payload).encode()),timeout=300)
            result=json.loads(stdout) if p.returncode==0 else {'metrics':None,'scorer_error':stderr.decode()[-2000:]}
        except asyncio.TimeoutError:
            p.kill();await p.wait()
            result={'metrics':None,'scorer_error':'Native metrics exceeded300seconds'}
        except ValueError as e:
            result={'metrics':None,'scorer_error':'Malformed scorer output: '+str(e)}
        metrics=result.get('metrics');value=metrics.get('argument_recall') if metrics else None
        extra={} if value is not None else {'mask_sample':True,'failure_kind':'bioplanner:scorer_or_reference_invalid','failure_reason':str(result)}
        return Response(**body.model_dump(by_alias=True),reward=value if value is not None else 0.0,metrics=metrics,scoring_diagnostics={k:v for k,v in result.items() if k!='metrics'},scoring_note='Reward is argument recall bookkeeping only; report all native metrics. Nulls are not zero. Provided definitions inflate function metrics.',**extra)
if __name__=='__main__':Server.run_webserver()
