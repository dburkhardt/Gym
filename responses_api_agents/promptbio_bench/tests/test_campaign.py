import pytest
from responses_api_agents.promptbio_bench.nemotron_proxy import MODEL, proxy, validate_result

def test_wrong_config_before_key_read(tmp_path):
    with pytest.raises(ValueError):
        with proxy(key_file=tmp_path/'missing',ledger=tmp_path/'ledger',model='gpt-5.4',base_url='https://api.openai.com/v1'): pass

@pytest.mark.parametrize('result',[
 {'model':'gpt-5.4'},
 {'model':MODEL,'choices':[{'finish_reason':'length','message':{'content':'yes'}}]},
 {'model':MODEL,'choices':[{'finish_reason':'stop','message':{'content':''}}]},
])
def test_reject_invalid_judge(result):
    with pytest.raises(ValueError):validate_result(result)

def test_completed_judge():
    validate_result({'model':MODEL,'choices':[{'finish_reason':'stop','message':{'content':'yes'}}]})

def test_full_native_campaign_config(tmp_path):
    from responses_api_agents.promptbio_bench.app import PromptBioConfig
    from responses_api_agents.promptbio_bench.nemotron_proxy import GATEWAY
    values=dict(host='127.0.0.1',port=40001,entrypoint='app.py',name='promptbio_agent',model_name=MODEL,model_base_url=GATEWAY,model_key_file=tmp_path/'key',sandbox_image='sha256:test',cache_dir=tmp_path/'cache',output_dir=tmp_path/'out',scorer_dir=tmp_path/'scorer',scorer_python='python',judge_key_file=tmp_path/'key',budget_ledger=tmp_path/'receipts',judge_model=MODEL,judge_base_url=GATEWAY)
    assert PromptBioConfig(**values).judge_model==MODEL
    with pytest.raises(ValueError):PromptBioConfig(**{**values,'comparison_models':['gpt-5.4']})
    with pytest.raises(ValueError):PromptBioConfig(**{**values,'model_base_url':'https://api.openai.com/v1'})
