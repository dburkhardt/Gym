"""Native creator MOOSE-Chem stages, collected as a Gym agent trajectory."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4
from fastapi import Request,Response
from pydantic import PrivateAttr
from nemo_gym.base_resources_server import BaseRunRequest,BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig,SimpleResponsesAPIAgent
from nemo_gym.openai_utils import NeMoGymResponse,NeMoGymResponseCreateParamsNonStreaming
from benchmarks.moosechem.native import MODEL

class Config(BaseResponsesAPIAgentConfig):
    native_python:Path
    upstream_repo:Path
    output_root:Path
    frozen_tasks:Path
    import_existing:bool=False

class RunRequest(BaseRunRequest):
    verifier_metadata:dict

class VerifyResponse(BaseVerifyResponse):
    artifact_dir:str
    native_metrics:dict
    reward_metric:str='final_round_inspiration_top3_recall_not_hypothesis_quality'
    synthetic_summary_response:bool=True

class Agent(SimpleResponsesAPIAgent):
    config:Config
    _semaphore:asyncio.Semaphore=PrivateAttr(default_factory=lambda:asyncio.Semaphore(1))

    async def responses(self,request:Request,response:Response,body:NeMoGymResponseCreateParamsNonStreaming)->NeMoGymResponse:
        raise NotImplementedError('Use /run with frozen task identifier')

    def execute(self,task_id):
        out=self.config.output_root/f'task{task_id}'
        if out.exists():
            if not self.config.import_existing or not (out/'result.json').exists():
                raise ValueError('Existing native attempt must be reconciled, not rerun')
        else:
            out.mkdir(parents=True)
            with (out/'native.log').open('x') as stream:
                subprocess.run([str(self.config.native_python),'-u','-m','benchmarks.moosechem.native',
                    '--repo',str(self.config.upstream_repo),'--output',str(out),'--task-id',str(task_id)],
                    stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=14400)
        result=json.loads((out/'result.json').read_text());screen=json.loads((out/'screening.json').read_text())
        ratios=list(screen[1].values());assert len(ratios)==1 and len(ratios[0])==4
        reward=ratios[0][-1][1];assert 0<=reward<=1
        traces=[json.loads(x) for x in (out/'model-traces.jsonl').read_text().splitlines()]
        assert traces and all(t['request']['model']==MODEL for t in traces)
        assert all(t.get('response',{}).get('model')==MODEL for t in traces if 'response' in t)
        return out,result,screen,reward,len(traces)

    async def run(self,request:Request,body:RunRequest)->VerifyResponse:
        task_id=body.verifier_metadata['id'];tasks={r['id']:r for r in map(json.loads,self.config.frozen_tasks.read_text().splitlines())};source=tasks[task_id]
        expected=source['question']+'\n\n'+source['survey'];actual=body.responses_create_params.model_dump()['input']
        assert len(actual)==1 and actual[0]['role']=='user' and actual[0]['content']==expected
        async with self._semaphore:
            out,result,screen,reward,n=await asyncio.to_thread(self.execute,task_id)
        text=json.dumps({'scope':result['scope'],'native_evaluation':result['evaluation']})
        response=NeMoGymResponse.model_validate({'id':'resp_'+uuid4().hex,'object':'response','created_at':time.time(),
            'model':MODEL,'status':'completed','output':[{'id':'msg_'+uuid4().hex,'type':'message','role':'assistant',
            'status':'completed','content':[{'type':'output_text','annotations':[],'text':text}]}],
            'parallel_tool_calls':False,'tool_choice':'auto','tools':[]})
        return VerifyResponse(**body.model_dump(),response=response,reward=reward,artifact_dir=str(out),
            native_metrics={'retrieval_by_round':screen[1],'full_creator_evaluation':result['evaluation'],
                            'actual_model_calls':n,'judge_variant':'Nemotron3.5SuperVL_all_roles','imported_saved_execution':self.config.import_existing})

if __name__=='__main__':Agent.run_webserver()
