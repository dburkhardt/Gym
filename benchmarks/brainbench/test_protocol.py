# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from omegaconf import OmegaConf
from fastapi.testclient import TestClient
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.server_utils import ServerClient
from resources_servers.brainbench_likelihood.app import Config, Server

@pytest.mark.parametrize('benchmark',['brainbench'])
def test_full_configuration_and_resource_startup(benchmark,monkeypatch):
    monkeypatch.setenv('CAMPAIGN_UPSTREAM_REPO','/tmp/creator')
    monkeypatch.setenv('CAMPAIGN_SAMPLE_JSONL','/tmp/sample.jsonl')
    initial=OmegaConf.merge(GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,{'config_paths':[f'benchmarks/{benchmark}/config.yaml']})
    parser=GlobalConfigDictParser()
    global_config=parser.parse(GlobalConfigDictParserConfig(initial_global_config_dict=initial,skip_load_from_cli=True,skip_load_from_dotenv=True,offline=True))
    instances=parser.filter_for_server_instance_configs(global_config)
    resource=next(x for x in instances if x.get_server_ref().name==benchmark+'_resources')
    config=Config.model_validate({**resource.get_inner_run_server_config_dict(),'name':benchmark+'_resources'})
    client=ServerClient(head_server_config={'host':'127.0.0.1','port':23001},global_config_dict=global_config)
    server=Server(config=config,server_client=client)
    with TestClient(server.setup_webserver()) as web:
        response=web.get('/reverify_mode')
        assert response.status_code==200
        assert 'stateless' in response.text.lower()


def test_mean_nll_and_tie_controls():
    from benchmarks.brainbench.protocol import MODEL,mean_nll,score_pair
    r={'model':MODEL,'usage':{'prompt_tokens':3},'prompt_logprobs':[None,{'1':{'logprob':-1}}, {'2':{'logprob':-3}}]}
    assert mean_nll(r)==2
    assert score_pair([1,2],0)['reward']==1
    assert score_pair([1,2],1)['reward']==0
    assert score_pair([1,1],0)['reward']==0
    r['prompt_logprobs'][1]['3']={'logprob':-0.2}
    with pytest.raises(AssertionError):mean_nll(r)

async def test_agent_scores_two_full_prompts_without_gold(monkeypatch):
    import json
    from responses_api_agents.brainbench_likelihood.app import BrainBenchAgent
    from responses_api_agents.non_executing_simple_agent.app import NonExecutingSimpleAgentConfig
    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from benchmarks.brainbench.protocol import MODEL
    seen=[]
    class HTTPResponse:
        def raise_for_status(self):pass
        def json(self):return {'model':MODEL,'usage':{'prompt_tokens':2},'prompt_logprobs':[None,{'42':{'logprob':-2}}]}
    class Client:
        def __init__(self,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def post(self,url,**kwargs):seen.append(kwargs['json']);return HTTPResponse()
    monkeypatch.setattr('responses_api_agents.brainbench_likelihood.app.httpx.AsyncClient',Client)
    monkeypatch.setenv('PROTEIN_NVIDIA_KEY','unit-test-no-secret')
    cfg=NonExecutingSimpleAgentConfig(name='test',host='127.0.0.1',port=23002,entrypoint='app.py',resources_server={'type':'resources_servers','name':'resources'},model_server={'type':'responses_api_models','name':'policy_model'})
    agent=BrainBenchAgent(config=cfg,server_client=ServerClient(head_server_config={'host':'127.0.0.1','port':23001},global_config_dict=OmegaConf.create({})))
    body=NeMoGymResponseCreateParamsNonStreaming(input=[{'role':'user','content':json.dumps({'choices':['abstract original','abstract altered']})}])
    result=await agent.responses(None,None,body)
    assert json.loads(result.output_text)['mean_nll']==[2,2]
    assert result.usage.input_tokens==4 and result.usage.total_tokens==4
    assert [r['messages'][0]['content'] for r in seen]==['abstract original','abstract altered']
    assert all(r['prompt_logprobs']==0 and r['max_tokens']==1 for r in seen)

def test_real_collation_preserves_exact_five_ids(monkeypatch,tmp_path):
    import json
    from pathlib import Path
    from nemo_gym.train_data_utils import TrainDataProcessor
    sample=Path('../../outputs/broad-eval-campaign/brainbench/sample-5.jsonl').resolve()
    if not sample.exists():pytest.skip('Prepared pinned source sample not present')
    monkeypatch.setenv('CAMPAIGN_SAMPLE_JSONL',str(sample))
    from benchmarks.brainbench.prepare import prepare
    assert prepare().read_bytes()==sample.read_bytes()
    initial=OmegaConf.merge(GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,{'config_paths':['benchmarks/brainbench/config.yaml'],'output_dirpath':str(tmp_path),'mode':'train_preparation','should_download':False,'overwrite_metrics_conflicts':True})
    cfg=GlobalConfigDictParser().parse(GlobalConfigDictParserConfig(initial_global_config_dict=initial,skip_load_from_cli=True,skip_load_from_dotenv=True,offline=True))
    TrainDataProcessor().run(cfg)
    actual=[json.loads(x) for x in (tmp_path/'benchmark.jsonl').read_text().splitlines()]
    assert len(actual)==5 and {r['uuid'] for r in actual}=={json.loads(x)['uuid'] for x in sample.read_text().splitlines()}
