# SPDX-License-Identifier: Apache-2.0
"""Native GeneGPT retrieval loop, with manual-only GeneHop scores explicitly masked."""
import asyncio,ast,json,os,time,re
from pathlib import Path
from uuid import uuid4
from fastapi import Request,Response
from pydantic import PrivateAttr
from nemo_gym.base_resources_server import BaseRunRequest,BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig,SimpleResponsesAPIAgent
from nemo_gym.openai_utils import NeMoGymResponse,NeMoGymResponseCreateParamsNonStreaming
from benchmarks.genehop.native_loop import NativeLoop,MODEL

class Config(BaseResponsesAPIAgentConfig):
    upstream_repo:Path
    output_root:Path
    base_url:str='http://127.0.0.1:49180/v1'
    api_key_env:str='GENEHOP_API_KEY'
class RunRequest(BaseRunRequest):
    uuid:str
    verifier_metadata:dict
class VerifyResponse(BaseVerifyResponse):
    verifier_metadata:dict
    uuid:str
    artifact_dir:str
    native_status:str
    raw_creator_score:float|None=None
    reward_metric:str='creator_disease_location_recall_only'
class GeneHopAgent(SimpleResponsesAPIAgent):
    config:Config
    _lock:asyncio.Semaphore=PrivateAttr(default_factory=lambda:asyncio.Semaphore(1))
    async def responses(self,request:Request,response:Response,body:NeMoGymResponseCreateParamsNonStreaming):
        raise NotImplementedError('Use native /run')
    def execute(self,body):
        if not re.fullmatch(r'[A-Za-z0-9_: -]+',body.uuid):raise ValueError('Unsafe task UUID')
        taskid=body.uuid.replace(':','-').replace(' ','_');out=self.config.output_root/taskid
        out.mkdir(parents=True,exist_ok=False)
        try:
            loop=NativeLoop(self.config.upstream_repo,out,os.environ[self.config.api_key_env],self.config.base_url)
            loop.header_path=self.config.output_root/'prompt-header.txt'
            result=loop.run(body.verifier_metadata['question'])
        except Exception as exc:
            result={'status':'infrastructure_failure','answer':None,'error':type(exc).__name__+': '+str(exc),'calls':[]}
            (out/'failure.json').write_text(json.dumps(result,indent=2))
        (out/'native-result.json').write_text(json.dumps(result,indent=2))
        return out,result
    async def run(self,request:Request,body:RunRequest):
        question=body.verifier_metadata['question'];inp=body.responses_create_params.model_dump()['input']
        if len(inp)!=1 or inp[0]['role']!='user' or inp[0]['content']!=question:raise ValueError('Question mismatch')
        async with self._lock:out,result=await asyncio.to_thread(self.execute,body)
        score=None;failure=None
        if result['status']!='completed':failure='genehop_native:'+result['status']
        elif body.verifier_metadata['category']!='Disease gene location':failure='genehop_native:manual_grading_required'
        else:
            tree=ast.parse((self.config.upstream_repo/'evaluate.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_answer');ns={}
            exec(compile(ast.Module(body=[node],type_ignores=[]),'creator-answer-parser','exec'),ns)
            answer=ns['get_answer'](result['answer'],'Disease gene location');gold=body.verifier_metadata['answer'];score=sum(x in answer for x in gold)/len(gold)
        response=NeMoGymResponse.model_validate({'id':'resp_'+uuid4().hex,'object':'response','created_at':time.time(),'model':MODEL,'status':'completed','output':[{'type':'message','id':'msg_'+uuid4().hex,'role':'assistant','status':'completed','content':[{'type':'output_text','text':result['answer'] or '', 'annotations':[]}]}],'parallel_tool_calls':False,'tool_choice':'auto','tools':[]})
        return VerifyResponse(**body.model_dump(by_alias=True),response=response,reward=score if score is not None else 0.0,raw_creator_score=score,mask_sample=score is None,failure_kind=failure,failure_reason='Unscored; inspect native status and creator manual-grading coverage' if failure else None,artifact_dir=str(out),native_status=result['status'])
if __name__=='__main__':GeneHopAgent.run_webserver()
