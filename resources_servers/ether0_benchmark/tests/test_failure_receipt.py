import asyncio

import pytest

from resources_servers.ether0_benchmark import app
from resources_servers.superchem.tests.test_native_contracts import response


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["success", "nonzero", "timeout", "malformed", "oserror", "cancel"])
async def test_subprocess_verifier_preserves_response(monkeypatch, tmp_path, case):
    from types import SimpleNamespace

    class Provider:
        closed = False

        def __init__(self, **kwargs):
            pass

        async def create(self, spec):
            return SimpleNamespace(raw={"workspace": tmp_path})

        async def close(self, handle):
            self.closed = True

        async def exec(self, handle, command, **kwargs):
            assert kwargs["timeout_s"] == 120
            assert "<answer>CCO</answer>" in (tmp_path / "request.json").read_text()
            if case == "oserror":
                raise OSError("spawn failed")
            if case == "cancel":
                raise asyncio.CancelledError()
            return SimpleNamespace(
                return_code=1 if case == "nonzero" else 0,
                error_type="timeout" if case == "timeout" else None,
                stdout="invalid"
                if case == "malformed"
                else 'ETHER0_RESULT_JSON:{"reward":1,"diagnostics":{}}\nnative diagnostic',
                stderr="controlled failure",
            )

    provider = Provider()
    monkeypatch.setattr(app, "LocalProvider", lambda **kwargs: provider)
    body = app.Request.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Question"}]},
            "response": response("<answer>CCO</answer>"),
            "verifier_metadata": {"solution": "sol_eval!:!0"},
        }
    )
    if case == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await app.Verifier().verify(body)
    else:
        result = await app.Verifier().verify(body)
        assert result.response == body.response
        if case == "success":
            assert result.reward == 1 and not result.mask_sample
        else:
            assert result.mask_sample and result.failure_kind == "ether0_benchmark:grader_unavailable"
            assert result.diagnostics["reward_is_serialization_placeholder"]
    assert provider.closed


def test_missing_asset_fails_before_native_import(monkeypatch, tmp_path):
    import importlib.util
    from types import SimpleNamespace

    from resources_servers.ether0_benchmark.grading import grade

    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(tmp_path / "__init__.py"))
    )
    with pytest.raises(RuntimeError, match="Missing or mismatched pinned creator asset"):
        grade("CCO", solution="fg_eval!:!x")


def test_score_worker_keeps_creator_prints_off_json_stdout(monkeypatch, capsys):
    import io
    import json
    import runpy
    import sys
    import types
    from pathlib import Path

    fake = types.ModuleType("resources_servers.ether0_benchmark.grading")

    def noisy_grade(**payload):
        print("creator diagnostic")
        return {"reward": 1.0}

    fake.grade = noisy_grade
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"answer":"C"}'))
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "score_worker.py"), run_name="__main__")
    captured = capsys.readouterr()
    assert json.loads(captured.out.removeprefix("ETHER0_RESULT_JSON:")) == {"reward": 1.0}
    assert "creator diagnostic" in captured.err
