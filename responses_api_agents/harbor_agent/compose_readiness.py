# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in Docker exec readiness for images with disabled healthchecks.

Compose 5's ``up --wait`` can reject an image declaring HEALTHCHECK NONE.
Use Harbor's existing sequential readiness path, including its explicit egress
policy wait, rather than changing the image or inventing a passing healthcheck.
This is an explicit runtime variant, not automatic fallback after a failed run.
Do not select it for tasks requiring custom application healthchecks.
"""

from dataclasses import replace

from harbor.environments.docker.docker import DockerEnvironment
from harbor.environments.docker.runtime import ContainerRuntime, DOCKER_RUNTIME


class ExecReadyDockerEnvironment(DockerEnvironment):
    """Keep native Docker lifecycle; select native sidecar/exec readiness."""

    @classmethod
    def runtime(cls) -> ContainerRuntime:
        return replace(DOCKER_RUNTIME, supports_compose_wait=False)
