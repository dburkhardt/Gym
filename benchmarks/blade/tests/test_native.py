# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from benchmarks.blade.native_phase import MODEL, Transport


def response():
    return {
        "id": "control",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "answer"}}],
    }


def test_transport_headers_durable_response_and_poison(tmp_path, monkeypatch):
    calls = []

    def send(socket_path, payload, headers):
        calls.append(headers)
        assert (tmp_path / "calls/000000.request.json").exists()
        return response()

    monkeypatch.setattr("benchmarks.blade.native_phase.post_json", send)
    transport = Transport(
        "http://localhost/ng-rollout/1/v1", "stable", "digest", tmp_path / "calls", socket_path="/dummy", token="token"
    )
    assert transport.create(model=MODEL, messages=[]).id == "control"
    assert (tmp_path / "calls/000000.response.json").exists()
    assert calls[0]["x-nemo-gym-logical-call-id"] == "blade:stable:0"

    def failure(*args, **kwargs):
        calls.append("failed")
        raise TimeoutError("ambiguous")

    monkeypatch.setattr("benchmarks.blade.native_phase.post_json", failure)
    with pytest.raises(RuntimeError):
        transport.create(model=MODEL, messages=[])
    with pytest.raises(RuntimeError, match="no resend"):
        transport.create(model=MODEL, messages=[])
    assert len(calls) == 2
    assert (tmp_path / "calls/000001.failure.json").exists()


def test_transport_existing_phase_and_wrong_model_refused(tmp_path):
    transport = Transport(
        "http://localhost/ng-rollout/1/v1", "stable", "digest", tmp_path / "calls", socket_path="/dummy", token="token"
    )
    with pytest.raises(ValueError):
        transport.create(model="other")
    with pytest.raises(FileExistsError):
        Transport(
            "http://localhost/ng-rollout/1/v1",
            "stable",
            "digest",
            tmp_path / "calls",
            socket_path="/dummy",
            token="token",
        )


def test_native_metric_vector_preserved():
    path = Path(
        "../../outputs/broad-eval-campaign/harbor/agent-expansion/blade-runtime/results/crofoot/scorer-run2/eval_metrics.json"
    )
    if not path.exists():
        pytest.skip("Saved creator result unavailable")
    metrics = json.loads(path.read_text())
    assert metrics["status"] == [5]
    keys = [key for key in metrics if key.startswith(("hit_rate_", "average_coverage_"))]
    assert len(keys) == 10
    assert metrics["average_coverage_models"] == 0.5
    assert metrics["average_coverage_vspecs"] == 1 / 14


def test_native_harbor_configuration_and_task_packages(tmp_path, monkeypatch):
    from harbor.models.task.task import Task
    from omegaconf import OmegaConf

    from benchmarks.blade.harbor_native import BladeNativeAgent, BladeNativeVerifier
    from benchmarks.blade.prepare import stage
    from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig

    source = Path("../../outputs/broad-eval-campaign/harbor/agent-expansion/sources/121/behavioral-data-BLADE-6118fa8")
    if not source.exists():
        pytest.skip("Pinned BLADE source unavailable")
    destination = tmp_path / "data"
    stage(source, destination)
    for dataset in ("crofoot", "panda_nuts", "amtl"):
        task = Task(task_dir=destination / "tasks" / dataset)
        assert task.config.agent.timeout_sec == 3720
        assert task.config.environment.memory_mb == 128
        assert task.instruction.strip() == dataset
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"config_paths": ["benchmarks/blade/config.yaml"], "allow_openai_version_skew": True},
    )
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial, skip_load_from_cli=True, skip_load_from_dotenv=True, offline=True
        )
    )
    agent = config["harbor_agent"]["responses_api_agents"]["harbor_agent"]
    assert agent["harbor_agent_import_path"].endswith(":BladeNativeAgent")
    assert agent["harbor_verifier_import_path"].endswith(":BladeNativeVerifier")
    assert not BladeNativeAgent.__abstractmethods__
    assert not BladeNativeVerifier.__abstractmethods__


@pytest.mark.asyncio
async def test_phase_sandbox_scope_has_no_ip_network_or_solver_references(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from benchmarks.blade import harbor_native
    from benchmarks.blade.prepare import stage

    source = Path("../../outputs/broad-eval-campaign/harbor/agent-expansion/sources/121/behavioral-data-BLADE-6118fa8")
    if not source.exists():
        pytest.skip("Pinned BLADE source unavailable")
    root = tmp_path / "staged"
    stage(source, root)
    from benchmarks.blade.prepare import stage_tokenizer

    tokenizer = Path("../../outputs/broad-eval-campaign/foundation/blade-native/tokenizer/cl100k_base.tiktoken")
    if not tokenizer.exists():
        pytest.skip("Pinned tokenizer fixture unavailable")
    stage_tokenizer(tokenizer, root)
    monkeypatch.setattr(harbor_native, "get_global_config_dict", lambda: {"provider_dispatch_guard": True})
    instances = []

    class Sandbox:
        def __init__(self, config, spec):
            self.config, self.spec, self.stopped = config, spec, False
            instances.append(self)

        async def start(self):
            pass

        async def serialize(self):
            return {"sandbox_id": "owned-control"}

        async def stop(self):
            self.stopped = True

        async def exec(self, *args, **kwargs):
            return SimpleNamespace(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(harbor_native, "AsyncSandbox", Sandbox)
    submitted = tmp_path / "submitted"
    submitted.mkdir()
    for phase in ("solver", "scorer"):
        await harbor_native.run_phase(
            root=root,
            dataset="crofoot",
            phase=phase,
            endpoint="http://localhost/ng-rollout/task/v1",
            output=tmp_path / phase,
            submitted=submitted if phase == "scorer" else None,
        )
        instance = instances[-1]
        assert instance.stopped and instance.config["docker"]["create"]["network"] == "none"
        volumes = instance.spec.provider_options["volumes"]
        assert not any("/var/run/docker" in volume for volume in volumes)
        assert not any(volume.endswith(":/controller:ro") for volume in volumes)
        references = [volume for volume in volumes if "annotations.csv" in volume]
        assert bool(references) == (phase == "scorer")
        assert not any("annotations.csv" in volume for volume in volumes) if phase == "solver" else True
        assert any(volume.endswith(":/model-socket:ro") for volume in volumes)
        assert instance.spec.env["BLADE_TOKEN"] not in (tmp_path / phase / "lifecycle.json").read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [0, 1])
async def test_native_setup_creates_carrier_workspace(monkeypatch, tmp_path, code):
    from types import SimpleNamespace

    from benchmarks.blade import harbor_native

    monkeypatch.setattr(harbor_native, "validate_staged", lambda root: None)
    calls = []

    class Environment:
        async def exec(self, command, timeout_sec):
            calls.append((command, timeout_sec))
            return SimpleNamespace(return_code=code)

    agent = object.__new__(harbor_native.BladeNativeAgent)
    agent.root = tmp_path
    if code:
        with pytest.raises(RuntimeError, match="workspace setup"):
            await agent.setup(Environment())
    else:
        await agent.setup(Environment())
    assert calls == [("mkdir -p /workspace", 30)]
