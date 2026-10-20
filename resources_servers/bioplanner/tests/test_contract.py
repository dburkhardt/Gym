import pytest,asyncio
from resources_servers.bioplanner.app import Request,Server
def response(text):
    return {'id':'fixture','created_at':0,'model':'nvidia/nvidia/nemotron-3.5-super-vl','object':'response','output':[{'id':'msg','type':'message','role':'assistant','status':'completed','content':[{'type':'output_text','text':text,'annotations':[]}]}],'parallel_tool_calls':False,'tool_choice':'auto','tools':[],'status':'completed'}
@pytest.mark.asyncio
async def test_partial_null_metrics_preserved(monkeypatch):
    monkeypatch.setenv('BIOPLANNER_SCORER_PYTHON','python')
    class Process:
        returncode=0
        async def communicate(self,data):return b'{"metrics":{"argument_recall":0.0,"argument_precision":null,"function_precision":1.0},"metric_errors":{}}',b''
    async def create(*args,**kwargs):return Process()
    monkeypatch.setattr(asyncio,'create_subprocess_exec',create)
    request=Request.model_validate({'id':'test','responses_create_params':{'input':[{'role':'user','content':'question'}]},'response':response(''),'verifier_metadata':{'gold_pseudocode':'','admissible_functions':''}})
    # Method is independent of server constructor/config for this contract.
    result=await Server.verify(None,request)
    assert result.metrics['argument_precision'] is None
    assert result.reward==0 and not result.mask_sample
    assert result.response==request.response
