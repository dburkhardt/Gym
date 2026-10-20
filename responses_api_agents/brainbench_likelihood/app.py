import json,os,time,uuid
import httpx
from fastapi import Request,Response
from nemo_gym.base_responses_api_agent import Body
from nemo_gym.openai_utils import NeMoGymResponse,NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.non_executing_simple_agent.app import NonExecutingSimpleAgent
from benchmarks.brainbench.protocol import MODEL,mean_nll
class BrainBenchAgent(NonExecutingSimpleAgent):
    async def responses(self,request:Request,response:Response,body:NeMoGymResponseCreateParamsNonStreaming=Body())->NeMoGymResponse:
        inputs=body.input
        assert len(inputs)==1
        content=inputs[0].content
        if not isinstance(content,str): raise ValueError('Expected JSON pair string')
        pair=json.loads(content);assert set(pair)=={'choices'} and len(pair['choices'])==2
        receipts=[];nll=[]
        async with httpx.AsyncClient(timeout=180) as client:
            for text in pair['choices']:
                payload={'model':MODEL,'messages':[{'role':'user','content':text}],'max_tokens':1,'prompt_logprobs':0,'stream':False}
                r=await client.post('http://127.0.0.1:49180/v1/chat/completions',json=payload,headers={'Authorization':'Bearer '+os.environ['PROTEIN_NVIDIA_KEY']});r.raise_for_status();data=r.json()
                nll.append(mean_nll(data));receipts.append({'request':payload,'response':data})
        return NeMoGymResponse.model_validate({'id':'brainbench-'+uuid.uuid4().hex,'created_at':time.time(),'model':MODEL,'object':'response','status':'completed','parallel_tool_calls':False,'tool_choice':'auto','tools':[],'output':[{'id':'likelihood','type':'message','role':'assistant','status':'completed','content':[{'type':'output_text','text':json.dumps({'mean_nll':nll,'receipts':receipts}),'annotations':[]}]}]})
if __name__=='__main__':BrainBenchAgent.run_webserver()
