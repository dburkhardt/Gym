# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from harbor.environments.docker.docker import DockerEnvironment
from harbor.environments.docker.runtime import DOCKER_RUNTIME

from responses_api_agents.harbor_agent.compose_readiness import ExecReadyDockerEnvironment


def test_only_readiness_capability_changes():
    native = DOCKER_RUNTIME
    selected = ExecReadyDockerEnvironment.runtime()
    assert selected.engine == native.engine
    assert selected.compose == native.compose
    assert selected.supports_compose_cp == native.supports_compose_cp
    assert selected.supports_compose_project_directory == native.supports_compose_project_directory
    assert selected.up_args == ["--detach"]
    assert native.up_args == ["--detach", "--wait"]
    assert ExecReadyDockerEnvironment.start is DockerEnvironment.start
    assert ExecReadyDockerEnvironment.stop is DockerEnvironment.stop


@pytest.mark.asyncio
@pytest.mark.parametrize("sidecar_fails", [False, True])
async def test_actual_native_start_gates_main_on_sidecar(monkeypatch, tmp_path, sidecar_fails):
    # Exercise the pinned Harbor start implementation rather than a recreated
    # sequence. Only external engine/build/mount calls are replaced.
    env = object.__new__(ExecReadyDockerEnvironment)
    monkeypatch.setattr(ExecReadyDockerEnvironment, "stream_enabled", property(lambda self: False))
    env.environment_dir = tmp_path
    env.task_env_config = SimpleNamespace(docker_image="immutable-image")
    env._enable_egress_control = True
    env._is_windows_container = False
    events = []
    import harbor.environments.docker.docker as native

    monkeypatch.setattr(native, "should_use_prebuilt_docker_image", lambda *a, **k: True)
    for name in [
        "_write_mounts_compose_file",
        "_write_env_compose_file",
        "_write_egress_control_services_compose_file",
        "_validate_daemon_mode",
    ]:
        monkeypatch.setattr(env, name, lambda: None)
    for name in [
        "_ensure_egress_control_sidecar_image_built",
        "_write_resources_compose_file",
        "_validate_image_os",
        "ensure_dirs",
        "_upload_environment_dir_after_start",
    ]:
        monkeypatch.setattr(env, name, AsyncMock())
    monkeypatch.setattr(env, "_mount_targets", lambda **k: [])

    async def compose(argv):
        events.append(argv)

    async def sidecar():
        events.append("sidecar-policy-ready")
        if sidecar_fails:
            raise RuntimeError("policy not ready")

    async def ready():
        events.append("main-exec-ready")

    monkeypatch.setattr(env, "_run_docker_compose_command", compose)
    monkeypatch.setattr(env, "_wait_until_egress_sidecar_ready", sidecar)
    monkeypatch.setattr(env, "_wait_until_compose_exec_ready", ready)
    if sidecar_fails:
        with pytest.raises(RuntimeError, match="policy not ready"):
            await env.start(False)
        assert ["up", "--detach"] not in events
        assert "main-exec-ready" not in events
    else:
        await env.start(False)
        assert events == [
            ["down", "--remove-orphans"],
            ["up", "--detach", env._EGRESS_CONTROL_SERVICE_NAME],
            "sidecar-policy-ready",
            ["up", "--detach"],
            "main-exec-ready",
        ]
