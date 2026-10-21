# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verifiers runtime contract backed by Gym's owned Docker sandbox.

The creator still owns orchestration, setup, artifact export and scoring. This
adapter retains outbound scientific networking; it never uses host networking.
"""

from __future__ import annotations

import asyncio
import re
import shlex
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from verifiers.v1.runtimes.base import Runtime

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.lifecycle import _owned_sandbox


def runtime_class(
    *,
    workspace: Path,
    source_dir: Path,
    receipt: Path,
    image: str,
    ttl_s: int = 25380,
    bridge_dir: Path | None = None,
    runtime_types=None,
) -> type[Runtime]:
    """Create a single-task native Runtime class without a second orchestration loop."""
    if runtime_types is None:
        from verifiers.v1.runtimes.base import ProgramResult, Runtime
        from verifiers.v1.runtimes.docker import DockerRuntimeInfo
    else:
        Runtime, ProgramResult, DockerRuntimeInfo = runtime_types

    if not re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", image):
        raise ValueError("An immutable qualified BioAgent image is required")
    for root in (workspace, source_dir):
        if root.is_symlink() or not root.is_absolute():
            raise ValueError("Runtime roots must be absolute and not symlinks")
    public = [source_dir / name for name in ("data", "reference") if (source_dir / name).exists()]
    for root in public:
        if root.is_symlink() or any(p.is_symlink() for p in root.rglob("*")):
            raise ValueError("Public asset mounts must not contain symlinks")
    if not (source_dir / "data").is_dir():
        raise ValueError("Task public data is unavailable")

    class ManagedRuntime(Runtime):
        # Interception reachability is supplied separately by the scoped UDS relay.
        # Claiming locality here would incorrectly route the container to host loopback.
        is_local = False

        def __init__(self, config, name=None):
            super().__init__(name)
            self.config = config
            self.info = DockerRuntimeInfo(**{**config.model_dump(), "image": image})
            if config.gpu:
                raise ValueError("BioAgent frozen CPU runtime does not permit GPU substitution")
            self._owner = None
            self.sandbox = AsyncSandbox(
                {
                    "docker": {
                        "create": {
                            "network": "bridge",
                            "pids_limit": 512,
                        }
                    }
                },
                SandboxSpec(
                    image=image,
                    workdir=config.workdir,
                    ttl_s=ttl_s,
                    resources=SandboxResources(cpu=config.cpu or 2, memory_mib=int((config.memory or 8) * 1024)),
                    metadata={"workload": "gym-bioagent", "run": self.name},
                    provider_options={
                        "volumes": [
                            f"{workspace}:{config.workdir}",
                            *(f"{path}:{path}" for path in public),
                            *([f"{bridge_dir}:/gym-bridge:ro"] if bridge_dir else []),
                        ]
                    },
                ),
            )

        async def start(self) -> None:
            self._owner = _owned_sandbox(self.sandbox, receipt)
            await self._owner.__aenter__()
            try:
                descriptor = await self.sandbox.serialize()
                self.info.id = descriptor["sandbox_id"]
            except BaseException as error:
                try:
                    await self.teardown()
                except BaseException as cleanup:
                    error.add_note(f"Owned runtime cleanup failed: {cleanup}")
                raise

        async def teardown(self) -> None:
            if self._owner is not None:
                owner, self._owner = self._owner, None
                await owner.__aexit__(None, None, None)

        async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
            if not argv:
                raise ValueError("Empty native runtime command")
            command = shlex.join(["env", *(f"{key}={value}" for key, value in env.items()), *argv])
            result = await self.sandbox.exec(command, timeout_s=ttl_s - 180)
            if result.error_type:
                raise RuntimeError(f"Managed native runtime execution failed: {result.error_type}")
            return ProgramResult(exit_code=result.return_code, stdout=result.stdout or "", stderr=result.stderr or "")

        async def run_background(self, argv: list[str], env: dict[str, str], log: str) -> None:
            command = shlex.join(["env", *(f"{key}={value}" for key, value in env.items()), *argv])
            result = await self.run(["sh", "-c", f"{command} > {shlex.quote(log)} 2>&1 < /dev/null &"], {})
            if result.exit_code:
                raise RuntimeError("Native background service failed to start")

        async def read(self, path: str) -> bytes:
            # Use native shared file transfer, not base64 through a bounded exec log.
            with tempfile.TemporaryDirectory(prefix="bioagent-read-", dir="/tmp") as temporary:
                target = Path(temporary) / "artifact"
                remote = path if path.startswith("/") else self.config.workdir.rstrip("/") + "/" + path
                await self.sandbox.download(remote, target)
                if target.is_symlink() or not target.is_file():
                    raise ValueError("Native artifact transfer did not produce a regular file")
                return await asyncio.to_thread(target.read_bytes)

        async def write(self, path: str, data: bytes) -> None:
            remote = path if path.startswith("/") else self.config.workdir.rstrip("/") + "/" + path
            result = await self.run(["mkdir", "-p", str(Path(remote).parent)], {})
            if result.exit_code:
                raise RuntimeError("Cannot prepare native setup directory")
            with tempfile.TemporaryDirectory(prefix="bioagent-write-", dir="/tmp") as temporary:
                source = Path(temporary) / "setup"
                source.write_bytes(data)
                await self.sandbox.upload(source, remote)

    return ManagedRuntime
