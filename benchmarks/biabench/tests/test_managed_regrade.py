# SPDX-License-Identifier: Apache-2.0
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from benchmarks.biabench import managed_regrade as runner


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [0, 1])
async def test_managed_verifier_preserves_failure_and_cleans_up(tmp_path, monkeypatch, code):
    inputs = [tmp_path / name for name in ("source", "task", "predictions")]
    for path in inputs:
        path.mkdir()
    (inputs[2] / "saved.csv").write_text("a,b\n1,2\n")
    output = tmp_path / "attempt"
    closed = []
    specs = []

    class Sandbox:
        def __init__(self, config, spec):
            assert config["docker"]["create"]["network"] == "none"
            specs.append(spec)

        async def exec(self, command, timeout_s):
            assert "regrade_outcome.py" in command
            assert timeout_s == 3600
            if code == 0:
                (output / "outcome.json").write_text(
                    json.dumps({"prediction_sha256": runner.prediction_hashes(inputs[2]), "new_model_calls": 0})
                )
            return SimpleNamespace(return_code=code, error_type=None, stdout="saved log", stderr="diagnostic")

    @asynccontextmanager
    async def owned(sandbox, receipt):
        try:
            yield sandbox
        finally:
            closed.append(True)

    monkeypatch.setattr(runner, "AsyncSandbox", Sandbox)
    monkeypatch.setattr(runner, "_owned_sandbox", owned)
    kwargs = dict(source=inputs[0], task=inputs[1], predictions=inputs[2], output=output, image="sha256:" + "a" * 64)
    if code:
        with pytest.raises(RuntimeError, match="not a scientific zero"):
            await runner.regrade_managed(**kwargs)
    else:
        record = await runner.regrade_managed(**kwargs)
        assert record["new_model_calls"] == 0
    assert closed == [True]
    assert (output / "stdout.log").read_text() == "saved log"
    assert json.loads((output / "execution.json").read_text())["return_code"] == code
    volumes = specs[0].provider_options["volumes"]
    assert all(volume.endswith(":ro") for volume in volumes[:-1])
    assert volumes[-1].endswith(":/results")
    with pytest.raises(FileExistsError):
        await runner.regrade_managed(**kwargs)
    assert len(specs) == 1
