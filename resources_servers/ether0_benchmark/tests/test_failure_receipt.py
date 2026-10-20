import asyncio
import pytest
from resources_servers.ether0_benchmark import app
from resources_servers.superchem.tests.test_native_contracts import response

class Process:
    def __init__(self,code=0,stdout=b'{"reward":1,"diagnostics":{}}',timeout=False):
        self.returncode=code; self.stdout=stdout; self.timeout=timeout; self.killed=False
    async def communicate(self,payload):
        assert b'<answer>CCO</answer>' in payload
        if self.timeout: raise asyncio.TimeoutError()
        return self.stdout,b'controlled failure'
    def kill(self): self.killed=True
    async def wait(self): return self.returncode

@pytest.mark.asyncio
@pytest.mark.parametrize('case',['success','nonzero','timeout','malformed','oserror'])
async def test_subprocess_verifier_preserves_response(monkeypatch,case):
    process=Process(code=1 if case=='nonzero' else 0,stdout=b'invalid' if case=='malformed' else b'{"reward":1,"diagnostics":{}}',timeout=case=='timeout')
    async def create(*args,**kwargs):
        assert args[1:]==('-m','resources_servers.ether0_benchmark.score_worker')
        if case=='oserror': raise OSError('spawn failed')
        return process
    monkeypatch.setattr(asyncio,'create_subprocess_exec',create)
    body=app.Request.model_validate({'responses_create_params':{'input':[{'role':'user','content':'Question'}]},'response':response('<answer>CCO</answer>'),'verifier_metadata':{'solution':'sol_eval!:!0'}})
    result=await app.Verifier().verify(body)
    assert result.response==body.response
    if case=='success': assert result.reward==1 and not result.mask_sample
    else:
        assert result.mask_sample and result.failure_kind=='ether0_benchmark:grader_unavailable'
        assert result.diagnostics['reward_is_serialization_placeholder']
    if case=='timeout': assert process.killed

def test_missing_asset_fails_before_native_import(monkeypatch,tmp_path):
    import importlib.util
    from types import SimpleNamespace
    from resources_servers.ether0_benchmark.grading import grade
    monkeypatch.setattr(importlib.util,'find_spec',lambda name:SimpleNamespace(origin=str(tmp_path/'__init__.py')))
    with pytest.raises(RuntimeError,match='Missing or mismatched pinned creator asset'):
        grade('CCO',solution='fg_eval!:!x')
