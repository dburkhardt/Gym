# SPDX-License-Identifier: Apache-2.0
"""Exercise actual owned lifecycle around a deterministic local provider double, never Docker."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from benchmarks.biabench import managed_process as module


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["success", "failure", "cancel", "identity_mismatch", "start_failure"])
async def test_managed_process_owned_cleanup_and_identity(tmp_path, monkeypatch, mode):
    paths = [tmp_path / name for name in ("source", "task", "submission")]
    for path in paths:
        path.mkdir()
    (paths[2] / "saved").write_text("unchanged")
    output = tmp_path / "result"
    events = []
    specs = []
    monkeypatch.setattr(module, "validate", lambda *args: None)

    class Sandbox:
        def __init__(self, config, spec):
            assert config == {"docker": {"create": {"network": "none"}}}
            specs.append(spec)

        async def start(self):
            events.append("start")
            if mode == "start_failure":
                raise RuntimeError("start failed")

        async def serialize(self):
            return {"owned_id": "test-owned"}

        async def stop(self):
            await asyncio.sleep(0)
            events.append("stop")

        async def exec(self, command, timeout_s):
            assert events[-1] == "start"
            assert command == "python -m benchmarks.biabench.process --request /results/request.json"
            assert timeout_s == 3600
            request = json.loads((output / "request.json").read_text())
            assert request["socket_path"] == "/bridge/model.sock"
            assert request["source"] == "/creator" and request["task"] == "/task"
            if mode == "cancel":
                raise asyncio.CancelledError()
            if mode == "failure":
                return SimpleNamespace(return_code=1, error_type="process_failure", stdout="", stderr="retained")
            identity = json.loads((output / "identity.json").read_text())
            child = output / "score"
            child.mkdir()
            (child / "identity.json").write_text(
                json.dumps({"digest": identity["digest"] if mode != "identity_mismatch" else "wrong"})
            )
            (child / "result.json").write_text(
                '{"status":"completed","native_scores":{"result_score":0.79,"checklist_score":null}}'
            )
            return SimpleNamespace(return_code=0, error_type=None, stdout="completed", stderr="")

    @asynccontextmanager
    async def bridge(path, **kwargs):
        assert kwargs["namespace"] == "biabench_process"
        assert kwargs["expected_model"] == module.MODEL
        identity = json.loads((output / "identity.json").read_text())
        assert kwargs["digest"] == identity["digest"]
        assert kwargs["lineage"] == identity["lineage"]
        events.append("bridge-start")
        try:
            yield
        finally:
            events.append("bridge-stop")

    monkeypatch.setattr(module, "AsyncSandbox", Sandbox)
    monkeypatch.setattr(module, "bridge", bridge)
    kwargs = dict(
        source=paths[0],
        task=paths[1],
        submission=paths[2],
        output=output,
        image="sha256:" + "a" * 64,
        model_endpoint="http://gym/ng-rollout/trial/v1",
        model_name=module.MODEL,
    )
    if mode == "success":
        result = await module.score_managed(**kwargs)
        assert result["native_scores"]["checklist_score"] is None
    else:
        with pytest.raises((RuntimeError, ValueError, asyncio.CancelledError)):
            await module.score_managed(**kwargs)
    assert events == ["bridge-start", "start", "stop", "bridge-stop"]
    if mode == "start_failure":
        assert (output / "lifecycle.error.json").is_file()
        assert not (output / "lifecycle.json").exists()
    else:
        receipt = json.loads((output / "lifecycle.json").read_text())
        assert receipt["cleanup_confirmed"] is True
    assert not (output / "request.json").exists()
    assert (paths[2] / "saved").read_text() == "unchanged"
    assert specs[0].resources.memory_mib == 4096
    assert specs[0].resources.cpu == 2
    volumes = specs[0].provider_options["volumes"]
    assert all(v.endswith(":ro") for v in volumes if not v.endswith(":/results"))
    with pytest.raises(FileExistsError):
        await module.score_managed(**kwargs)
    assert len(specs) == 1


def test_config_admits_full_overlapping_memory():
    from pathlib import Path

    import yaml

    from benchmarks.biabench import harbor_prepare

    config = yaml.safe_load(Path(harbor_prepare.__file__).with_name("config.yaml").read_text())
    agent = config["harbor_agent"]["responses_api_agents"]["harbor_agent"]
    assert agent["harbor_ray_task_memory_bytes"] == 22 * 1024**3
    assert agent["concurrency"] == 1
    # Native prepared task retains8GiB in both solver and inherited separate verifier.
    source = Path(harbor_prepare.__file__).read_text()
    assert "memory_mb = 8192" in source
    assert 'environment_mode = "separate"' in source
