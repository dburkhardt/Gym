import pytest
from resources_servers.ether0_benchmark import app
from resources_servers.superchem.tests.test_native_contracts import response

@pytest.mark.asyncio
async def test_unavailable_oracle_preserves_model_response(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError('Creator oracle service unavailable')
    monkeypatch.setattr(app, 'grade', unavailable)
    body=app.Request.model_validate({'responses_create_params':{'input':[{'role':'user','content':'Question'}]},'response':response('<answer>CCO</answer>'),'verifier_metadata':{'solution':'sol_eval!:!0'}})
    result=await app.Verifier().verify(body)
    assert result.mask_sample is True
    assert result.failure_kind=='ether0_benchmark:grader_unavailable'
    assert result.diagnostics['reward_is_serialization_placeholder'] is True
    assert result.response==body.response
