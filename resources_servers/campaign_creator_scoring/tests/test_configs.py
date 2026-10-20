# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from omegaconf import OmegaConf
from fastapi.testclient import TestClient
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.server_utils import ServerClient
from resources_servers.campaign_creator_scoring.app import Config, CreatorResourcesServer

@pytest.mark.parametrize('benchmark',['qcbench','chemiq','sino_us_drugqa','moleculariq','fgbench','scienceqa','scibench'])
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
