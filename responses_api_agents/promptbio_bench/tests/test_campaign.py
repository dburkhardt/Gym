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
