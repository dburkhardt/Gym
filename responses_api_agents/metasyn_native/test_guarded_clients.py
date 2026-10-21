# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import httpx
import pytest
from openai import APIConnectionError
from responses_api_agents.metasyn_native.guarded_clients import guarded_clients, MODEL


def test_solver_and_judge_share_durable_guard_with_actual_sdk(tmp_path):
    rag, judge = SimpleNamespace(OpenAI=object), SimpleNamespace(OpenAI=object)
    seen = []
    async def provider(request):
        seen.append(request)
        return httpx.Response(200, json={'id':'call','object':'chat.completion','created':0,
            'model':MODEL,'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'answer'}}]})
    with guarded_clients(rag=rag,llm_client=judge,socket_path='unused',token='scoped',
        lineage='attempt',digest='digest',output=tmp_path/'calls',transport=httpx.MockTransport(provider)):
        for module in (rag,judge):
            result=module.OpenAI(api_key='not-forwarded',base_url='http://historical').chat.completions.create(
                model=MODEL,messages=[{'role':'user','content':'original prompt'}])
            assert result.choices[0].message.content=='answer'
    assert len(seen)==2
    assert seen[0].headers['authorization']=='Bearer scoped'
    assert (tmp_path/'calls/000001.response.json').exists()
    assert rag.OpenAI is object and judge.OpenAI is object


def test_creator_retry_after_timeout_cannot_forward_again(tmp_path):
    rag,judge=SimpleNamespace(OpenAI=object),SimpleNamespace(OpenAI=object)
    seen=[]
    async def provider(request):
        seen.append(request)
        raise httpx.ReadTimeout('ambiguous')
    with guarded_clients(rag=rag,llm_client=judge,socket_path='unused',token='scoped',
        lineage='attempt',digest='digest',output=tmp_path/'calls',transport=httpx.MockTransport(provider)):
        for _ in range(2):
            with pytest.raises(APIConnectionError):
                rag.OpenAI().chat.completions.create(model=MODEL,messages=[{'role':'user','content':'same'}])
    assert len(seen)==1
