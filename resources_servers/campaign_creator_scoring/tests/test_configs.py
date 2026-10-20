# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from omegaconf import OmegaConf
from fastapi.testclient import TestClient
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.server_utils import ServerClient
from resources_servers.campaign_creator_scoring.app import Config, CreatorResourcesServer

@pytest.mark.parametrize('benchmark',['qcbench','chemiq','sino_us_drugqa','moleculariq','fgbench','scienceqa','scibench','fukuyama','mollangbench','suprabench'])
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
    server=CreatorResourcesServer(config=config,server_client=client)
    with TestClient(server.setup_webserver()) as web:
        response=web.get('/reverify_mode')
        assert response.status_code==200
        assert 'stateless' in response.text.lower()


def test_verify_endpoint_with_pinned_qcbench(monkeypatch):
    import json
    from pathlib import Path
    repo=Path('../qcbench-upstream').resolve()
    root=Path('../../outputs/broad-eval-campaign').resolve()
    sample=root/'scientific-agents/qcbench/qcbench-5.jsonl'
    trace=root/'proteinlmbench/pilot-20-v2.jsonl'
    if not (repo.exists() and sample.exists() and trace.exists()):
        pytest.skip('Campaign pinned source and saved response fixture unavailable')
    monkeypatch.setenv('CAMPAIGN_UPSTREAM_REPO',str(repo))
    monkeypatch.setenv('CAMPAIGN_SAMPLE_JSONL',str(sample))
    initial=OmegaConf.merge(GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,{'config_paths':['benchmarks/qcbench/config.yaml']})
    parser=GlobalConfigDictParser()
    global_config=parser.parse(GlobalConfigDictParserConfig(initial_global_config_dict=initial,skip_load_from_cli=True,skip_load_from_dotenv=True,offline=True))
    resource=next(x for x in parser.filter_for_server_instance_configs(global_config) if x.get_server_ref().name=='qcbench_resources')
    config=Config.model_validate({**resource.get_inner_run_server_config_dict(),'name':'qcbench_resources'})
    server=CreatorResourcesServer(config=config,server_client=ServerClient(head_server_config={'host':'127.0.0.1','port':23001},global_config_dict=global_config))
    row=json.loads(sample.read_text().splitlines()[0]);response=json.loads(trace.read_text().splitlines()[0])['response']
    response['output']=[{'id':'control','type':'message','role':'assistant','status':'completed','content':[{'type':'output_text','text':'\\boxed{'+row['verifier_metadata']['answer']+'}','annotations':[]}]}]
    with TestClient(server.setup_webserver()) as web:
        result=web.post('/verify',json={**row,'response':response})
        assert result.status_code==200,result.text
        assert result.json()['reward']==1
        assert result.json()['scoring_details']['metric']=='creator_numeric'


@pytest.mark.parametrize('benchmark',['qcbench','chemiq','sino_us_drugqa','moleculariq','fgbench','scienceqa','scibench','fukuyama','mollangbench','suprabench'])
def test_prepare_and_real_collation(benchmark,monkeypatch,tmp_path):
    import importlib
    import json
    from pathlib import Path
    from nemo_gym.train_data_utils import TrainDataProcessor
    sample=Path('../../outputs/broad-eval-campaign/scientific-agents')/benchmark/(benchmark+'-5.jsonl')
    if not sample.exists():
        pytest.skip('Pinned campaign sample unavailable')
    monkeypatch.setenv('CAMPAIGN_SAMPLE_JSONL',str(sample.resolve()))
    monkeypatch.setenv('CAMPAIGN_UPSTREAM_REPO','/tmp/creator')
    staged=importlib.import_module('benchmarks.'+benchmark+'.prepare').prepare()
    assert staged.read_bytes()==sample.read_bytes()
    initial=OmegaConf.merge(GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,{'config_paths':[f'benchmarks/{benchmark}/config.yaml'],'output_dirpath':str(tmp_path),'mode':'train_preparation','should_download':False,'overwrite_metrics_conflicts':True})
    global_config=GlobalConfigDictParser().parse(GlobalConfigDictParserConfig(initial_global_config_dict=initial,skip_load_from_cli=True,skip_load_from_dotenv=True,offline=True))
    TrainDataProcessor().run(global_config)
    actual=[json.loads(x) for x in (tmp_path/'benchmark.jsonl').read_text().splitlines()]
    expected=[json.loads(x) for x in sample.read_text().splitlines()]
    assert len(actual)==len(expected)==5
    assert {x['uuid'] for x in actual}=={x['uuid'] for x in expected}
