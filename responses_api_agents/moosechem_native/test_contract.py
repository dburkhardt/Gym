import json
from types import SimpleNamespace
import pytest
from responses_api_agents.moosechem_native.app import Agent
from benchmarks.moosechem.native import MODEL


def fixture(tmp_path,model=MODEL):
    out=tmp_path/'task18';out.mkdir()
    (out/'result.json').write_text(json.dumps({'scope':'fixture','evaluation':[]}))
    (out/'screening.json').write_text(json.dumps([{}, {'question':[[1,1],[0.5,1],[0.5,0.5],[0,0.5]]}]))
    (out/'model-traces.jsonl').write_text(json.dumps({'request':{'model':MODEL},'response':{'model':model}})+'\n')
    return SimpleNamespace(config=SimpleNamespace(output_root=tmp_path,import_existing=True))


def test_existing_execution_import_preserves_native_component(tmp_path):
    stub=fixture(tmp_path)
    _,result,screen,reward,calls=Agent.execute(stub,18)
    assert reward==0.5 and calls==1 and result['evaluation']==[]


def test_different_model_trace_rejected(tmp_path):
    with pytest.raises(AssertionError):Agent.execute(fixture(tmp_path,'another-model'),18)


def test_existing_attempt_never_silently_repeated(tmp_path):
    stub=fixture(tmp_path);stub.config.import_existing=False
    with pytest.raises(ValueError,match='reconciled'):Agent.execute(stub,18)
