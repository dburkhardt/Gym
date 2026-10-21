# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from harbor.environments.docker.docker import DockerEnvironment

from responses_api_agents.harbor_agent.artifact_environment import BundledArtifactDockerEnvironment


@pytest.fixture
def environment(monkeypatch):
    env = object.__new__(BundledArtifactDockerEnvironment)
    env._bundle_sources = frozenset()
    env._bundle_python = sys.executable

    async def upload(source, target):
        shutil.copyfile(source, target)

    async def execute(command, **kwargs):
        proc = await asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await proc.communicate()
        return SimpleNamespace(return_code=proc.returncode, stdout=out.decode(), stderr=err.decode())

    monkeypatch.setattr(env, "upload_file", upload)
    monkeypatch.setattr(env, "download_file", upload)
    monkeypatch.setattr(env, "exec", execute)
    return env


@pytest.mark.asyncio
@pytest.mark.parametrize("service", [False, True])
async def test_readonly_nested_export(environment, tmp_path, service):
    source = tmp_path / "source"
    nested = source / "input/nested"
    nested.mkdir(parents=True)
    data = nested / "data"
    data.write_bytes(b"known scientific bytes")
    data.chmod(0o444)
    nested.chmod(0o555)
    nested.parent.chmod(0o555)
    environment._bundle_sources = frozenset([str(source)])
    target = tmp_path / "capture"
    target.mkdir()
    method = environment.service_download_dir if service else environment.download_dir
    await method(str(source), target)
    assert (target / "input/nested/data").read_bytes() == data.read_bytes()
    assert (target / "input/nested").stat().st_mode & 0o777 == 0o555
    assert (target / "input/nested/data").stat().st_mode & 0o777 == 0o444
    assert data.stat().st_mode & 0o777 == 0o444
    receipt = json.loads(target.with_name("capture.bundle-receipt.json").read_text())
    assert receipt["source"] == str(source)
    # Allow pytest to remove only this synthetic fixture afterwards.
    for p in [nested, nested.parent, target / "input/nested", target / "input"]:
        p.chmod(0o755)


@pytest.mark.asyncio
async def test_existing_partial_and_symlink_refuse_before_export(environment, tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    environment._bundle_sources = frozenset([str(source)])
    export = AsyncMock()
    monkeypatch.setattr(environment, "upload_file", export)
    target = tmp_path / "partial"
    target.mkdir()
    (target / "existing").write_text("preserved")
    with pytest.raises(FileExistsError):
        await environment.download_dir(str(source), target)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        await environment.download_dir(str(source), link)
    export.assert_not_called()
    assert (target / "existing").read_text() == "preserved"


@pytest.mark.asyncio
async def test_source_link_refuses_no_published_capture(environment, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "escape").symlink_to("/etc/passwd")
    environment._bundle_sources = frozenset([str(source)])
    with pytest.raises(RuntimeError, match="export failed"):
        await environment.download_dir(str(source), tmp_path / "capture")
    assert not (tmp_path / "capture").exists()


@pytest.mark.asyncio
async def test_unselected_delegates_and_selected_services_exclusions_refuse(environment, monkeypatch, tmp_path):
    direct = AsyncMock()
    service = AsyncMock()
    monkeypatch.setattr(DockerEnvironment, "download_dir", direct)
    monkeypatch.setattr(DockerEnvironment, "service_download_dir", service)
    await environment.download_dir("/other", tmp_path / "a")
    direct.assert_awaited_once()
    await environment.service_download_dir("/other", tmp_path / "b", service="aux")
    service.assert_awaited_once()
    environment._bundle_sources = frozenset(["/required"])
    with pytest.raises(ValueError, match="main service"):
        await environment.service_download_dir("/required", tmp_path / "c", service="aux")
    with pytest.raises(ValueError, match="exclusions"):
        await environment.service_download_dir_with_exclusions("/required", tmp_path / "d", ["secret"])


@pytest.mark.parametrize("sources", [[], ["/"], ["relative"], ["/a/../b"], ["/a//b"]])
def test_invalid_source_paths(sources):
    with pytest.raises(ValueError, match="normalized absolute"):
        BundledArtifactDockerEnvironment(bundle_artifact_sources=sources)


@pytest.mark.asyncio
async def test_actual_artifact_handler_manifest(environment, tmp_path, monkeypatch):
    import logging
    from harbor.models.trial.config import ArtifactConfig
    from harbor.trial.artifact_handler import ArtifactHandler

    source = tmp_path / "source"
    source.mkdir()
    (source / "result").write_bytes(b"preserved")
    environment._bundle_sources = frozenset([str(source)])
    monkeypatch.setattr(environment, "service_is_dir", AsyncMock(return_value=True))
    handler = ArtifactHandler(artifacts=[], logger=logging.getLogger(__name__))
    entry = await handler._download_artifact(
        source_env=environment,
        artifacts_dir=tmp_path / "artifacts",
        artifact=ArtifactConfig(source=str(source), destination="workspace"),
        convention_source="/logs/artifacts",
    )
    assert entry.status == "ok"
    assert entry.source == str(source)
    assert entry.destination == "artifacts/workspace"
    assert (tmp_path / "artifacts/workspace/result").read_bytes() == b"preserved"
    # Existing required-artifact failure policy remains authoritative on failure.
    from responses_api_agents.harbor_agent.failure_diagnostics import required_artifact_failure

    (source / "link").symlink_to("/etc/passwd")
    second = ArtifactHandler(artifacts=[], logger=logging.getLogger(__name__))
    failed = await second._download_artifact(
        source_env=environment,
        artifacts_dir=tmp_path / "failure",
        artifact=ArtifactConfig(source=str(source), destination="workspace"),
        convention_source="/logs/artifacts",
    )
    assert failed.status == "failed"
    assert required_artifact_failure({"entries": [failed.model_dump()]}, [str(source)])[0] is not None
