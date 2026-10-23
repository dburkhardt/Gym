# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise installed Harbor collection, not a replacement artifact exporter."""

import json
import logging
import shutil
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest


pytest.importorskip("harbor")
from harbor.models.task.config import ArtifactConfig, TaskConfig
from harbor.trial.artifact_handler import ArtifactHandler
from harbor.trial.single_step import SingleStepTrial

from responses_api_agents.harbor_agent.tests.test_app import _make_server


def job(server):
    return server._build_job_config(
        "scientific", "test_task_123", "model", "http://127.0.0.1:1/v1", "artifacts", Path("/tmp/artifacts")
    )


def test_upstream_artifact_config_is_forwarded_and_default_unchanged():
    assert job(_make_server())["artifacts"] == []
    entries = ["/workspace", {"source": "/logs/custom", "destination": "diagnostics", "exclude": ["*.tmp"]}]
    actual = job(_make_server(harbor_artifacts=entries))["artifacts"]
    assert actual[0] == "/workspace"
    assert actual[1] == ArtifactConfig(**entries[1]).model_dump(mode="json")
    with pytest.raises(ValueError):
        job(_make_server(harbor_artifacts=[{"source": "/workspace/../reference"}]))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [SystemExit, RuntimeError])
async def test_installed_trial_collects_workspace_before_verifier_constructor_failure(tmp_path, failure):
    workspace = tmp_path / "container-workspace"
    workspace.mkdir()
    (workspace / "report.md").write_bytes(b"partial submitted report\x00\n")
    reference = tmp_path / "judge-only"
    reference.mkdir()
    (reference / "secret.json").write_text("private judge fixture")
    calls = []

    class Environment:
        capabilities = SimpleNamespace(mounted=False)

        async def service_is_dir(self, source, **kwargs):
            return True

        async def service_download_dir(self, *, source_dir, target_dir, service):
            calls.append(source_dir)
            if source_dir == "/workspace":
                shutil.copytree(workspace, target_dir, dirs_exist_ok=True)
            elif source_dir != "/logs/artifacts":
                raise AssertionError("Unexpected reference or source access")

    # Run installed SingleStepTrial._run and its real phased ArtifactHandler.
    # Only agent execution and verifier construction are fixture boundaries.
    trial = object.__new__(SingleStepTrial)
    trial.task = SimpleNamespace(config=TaskConfig())
    trial.paths = SimpleNamespace(artifacts_dir=tmp_path / "artifacts")
    trial.agent_env_paths = SimpleNamespace(artifacts_dir=Path("/logs/artifacts"))
    trial.agent_environment = Environment()
    trial.logger = logging.getLogger("artifact-preservation-test")
    trial._artifact_handler = ArtifactHandler(artifacts=["/workspace"], logger=trial.logger)
    trial._are_artifacts_collected = False
    trial._run_agent = AsyncMock()
    trial._upload_agent_logs = AsyncMock()

    class FailingVerifier:
        def __init__(self):
            assert (trial.paths.artifacts_dir / "workspace/report.md").read_bytes() == (
                workspace / "report.md"
            ).read_bytes()
            raise failure(2)

    async def construct(self):
        FailingVerifier()

    trial._run_verifier = MethodType(construct, trial)
    with pytest.raises(failure):
        await trial._run()
    assert calls == ["/logs/artifacts", "/workspace"]
    assert not (trial.paths.artifacts_dir / "judge-only").exists()
    assert (reference / "secret.json").read_text() == "private judge fixture"
    manifest = json.loads((trial.paths.artifacts_dir / "manifest.json").read_text())
    assert next(x for x in manifest if x["source"] == "/workspace")["status"] == "ok"
    # Re-collection is idempotent and does not overwrite the retained submission.
    await trial._collect_artifacts()
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_upstream_destination_collision_preserves_first_capture(tmp_path):
    class Environment:
        capabilities = SimpleNamespace(mounted=False)

        async def service_is_dir(self, source, **kwargs):
            return True

        async def service_download_dir(self, *, source_dir, target_dir, service):
            assert source_dir == "/workspace"
            (target_dir / "report").write_text("original")

    handler = ArtifactHandler(
        artifacts=[
            ArtifactConfig(source="/workspace", destination="saved"),
            ArtifactConfig(source="/other", destination="saved"),
        ],
        logger=logging.getLogger("collision"),
    )
    handler.begin_collection()
    await handler.download_artifacts(Environment(), tmp_path, source_artifacts_dir=Path("/logs/artifacts"))
    assert (tmp_path / "saved/report").read_text() == "original"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert next(x for x in manifest if x["source"] == "/other")["status"] == "skipped"


def test_researchclaw_declares_only_submitted_workspace():
    import yaml

    config = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text())
    agent = config["harbor_agent"]["responses_api_agents"]["harbor_agent"]
    assert agent["harbor_artifacts"] == ["/workspace"]
    assert agent["harbor_required_artifact_sources"] == ["/workspace"]
