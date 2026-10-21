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


@pytest.mark.parametrize("metadata,expected", [
    ({"_ng_task_index": 18, "_ng_rollout_index": 0}, "18-0"),
    ({"_ng_rollout_id": "campaign-task18"}, "campaign-task18"),
])
def test_native_collection_identity_survives_request_validation(metadata, expected):
    from responses_api_agents.moosechem_native.app import RunRequest
    from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
    body = RunRequest.model_validate({
        "responses_create_params": {"input": []},
        "verifier_metadata": {"id": 18}, **metadata,
    })
    assert maybe_rollout_id_from_run_body(body) == expected
    for key, value in metadata.items():
        if key == "_ng_rollout_id":
            assert body.capture_rollout_id == value  # Upstream deliberately excludes this routing-only field.
        else:
            assert body.model_dump()[key] == value


@pytest.mark.asyncio
async def test_saved_native_result_preserves_collection_identity(tmp_path):
    import asyncio
    from responses_api_agents.moosechem_native.app import RunRequest
    stub = fixture(tmp_path)
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"id":18,"question":"Question","survey":"Survey"})+"\n")
    stub.config.frozen_tasks = tasks
    stub._semaphore = asyncio.Semaphore(1)
    stub.read_result = lambda task: Agent.read_result(stub, task)
    body = RunRequest.model_validate({
        "responses_create_params":{"input":[{"role":"user","content":"Question\n\nSurvey"}]},
        "verifier_metadata":{"id":18},
        "_ng_task_index":18,"_ng_rollout_index":0,"_ng_attempt_index":1,
    })
    result = await Agent.run(stub, None, body)
    serialized = result.model_dump()
    for key in ("_ng_task_index", "_ng_rollout_index", "_ng_attempt_index"):
        assert serialized[key] == body.model_dump()[key]
    assert result.reward == 0.5
