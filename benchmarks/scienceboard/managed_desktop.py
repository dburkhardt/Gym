# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Creator desktop interface backed by Gym ownership; Linux bridge runtime remains gated."""

import ipaddress
import time
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import requests

from benchmarks.scienceboard.admission import DIGEST
from nemo_gym.sandbox import Sandbox, SandboxResources, SandboxSpec


class ManagedDesktop:
    """Implement the pinned VManager's desktop contract with one owned sandbox.

    A creator runner supplies its unmodified DockerController class. No task/reference
    mount, host network, credential environment or published control port is permitted.
    The image's native entrypoint is preserved. Its custom entrypoint means Docker
    provider TTL is unavailable: the owning creator process must always call close.
    """

    server_port = 5000

    def __init__(
        self,
        image: str,
        *,
        controller_factory: Callable,
        snapshot_name: str = "sci_bench",
        startup_timeout: float = 240,
        sandbox_factory: Callable = Sandbox,
        session_factory: Callable = requests.Session,
        receipt_dir: Path | None = None,
    ) -> None:
        if not DIGEST.fullmatch(image) or snapshot_name != "sci_bench" or startup_timeout <= 0:
            raise ValueError("An immutable admitted image and native sci_bench snapshot are required")
        self.image = self.path_to_vm = image
        self.snapshot_name = snapshot_name
        self.startup_timeout = startup_timeout
        self.controller_factory = controller_factory
        self.sandbox_factory = sandbox_factory
        self.session_factory = session_factory
        self.sandbox = None
        self.controller = None
        self.vm_ip = None
        self._http = None
        self.receipt_dir = receipt_dir
        self._start_emulator()

    def _start_emulator(self) -> None:
        if self.sandbox is not None:
            return
        self._http = self.session_factory()
        self._http.trust_env = False
        run_id = uuid4().hex
        if self.receipt_dir is not None:
            from nemo_gym.sandbox.guarded_chat_transport import publish

            self.receipt_dir.mkdir(parents=True, exist_ok=True)
            publish(
                self.receipt_dir / f"{run_id}.intent.json",
                {"ownership_labels": {"nemo-gym.run": run_id, "nemo-gym.workload": "scienceboard-desktop"}},
            )
        try:
            self.sandbox = self.sandbox_factory(
                {"docker": {"create": {"network": "bridge", "use_init": True}}},
                SandboxSpec(
                    image=self.image,
                    entrypoint=["/usr/local/bin/scienceboard-entrypoint"],
                    resources=SandboxResources(cpu=8, memory_mib=24576),
                    metadata={"workload": "scienceboard-desktop", "run": run_id},
                    env={"DISPLAY": ":0", "LIBGL_ALWAYS_SOFTWARE": "1", "GALLIUM_DRIVER": "llvmpipe"},
                    provider_options={
                        "run_args": ["--pull", "never", "--hostname", "scienceboard", "--shm-size", "2g"]
                    },
                ),
            )
            self.sandbox.start()
            if self.receipt_dir is not None:
                from nemo_gym.sandbox.guarded_chat_transport import publish

                self.receipt_dir.mkdir(parents=True, exist_ok=True)
                publish(self.receipt_dir / f"{run_id}.descriptor.json", self.sandbox.serialize())
            result = self.sandbox.exec("hostname -i", timeout_s=10)
            if result.return_code != 0:
                raise RuntimeError("Desktop bridge address unavailable")
            address = result.stdout.strip()
            parsed = ipaddress.ip_address(address)
            if parsed.version != 4 or parsed.is_loopback or parsed.is_unspecified or not parsed.is_private:
                raise RuntimeError("Expected one private Linux Docker bridge IPv4 address")
            self.vm_ip = address
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                try:
                    health = self._http.get(f"http://{address}:5000/health", timeout=2)
                    if health.status_code == 200:
                        shot = self._http.get(f"http://{address}:5000/screenshot", timeout=10)
                        if shot.status_code == 200 and shot.content.startswith(b"\x89PNG"):
                            self.controller = self.controller_factory(vm_ip=address, server_port=5000)
                            return
                except requests.RequestException:
                    pass
                time.sleep(1)
            raise TimeoutError("Creator desktop health/screenshot readiness timed out")
        except BaseException as original:
            try:
                self.close()
            except BaseException as cleanup:
                original.add_note(f"Owned desktop cleanup also failed: {type(cleanup).__name__}")
            raise

    def _revert_to_snapshot(self) -> str:
        if self.snapshot_name != "sci_bench":
            raise ValueError("Unknown creator snapshot")
        self.close()
        return self.image

    def close(self) -> None:
        """Remove only the owned sandbox; retain its handle on cleanup failure."""
        try:
            if self.sandbox is not None:
                self.sandbox.stop()
                self.sandbox = None
                self.controller = None
                self.vm_ip = None
        finally:
            if self._http is not None:
                self._http.close()
                self._http = None


def creator_desktop_factory(
    source_root: Path, *, metadata: dict, audit: dict, resources: dict, receipt_dir: Path | None = None
) -> Callable:
    """Return a pinned VManager-compatible constructor after image evidence admission.

    Install only in an isolated creator worker (`sci.docker.DockerDesktopEnv = factory`)
    before constructing Tester; retain creator Tester, agent, tasks and verifiers.
    This function itself neither launches Docker nor changes a module global.
    """
    import importlib
    import json
    import os

    from benchmarks.scienceboard.admission import admit
    from benchmarks.scienceboard.prepare import MANIFEST, checked_source

    root = Path(source_root).resolve()
    receipt = admit(metadata, audit, resources)
    if receipt["blockers"]:
        raise ValueError("ScienceBoard image evidence has not passed admission")
    # Creator application clients use module-level requests; until a qualified
    # per-controller adapter exists, don't silently send desktop traffic to proxies.
    if any(
        os.environ.get(key)
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    ):
        raise ValueError("Creator desktop worker requires proxy-free environment")
    manifest = json.loads(MANIFEST.read_text())
    for relative, expected in manifest["files"].items():
        checked_source(root, relative, expected)
    module = importlib.import_module("sci.docker.controller")
    if Path(module.__file__).resolve() != root / "sci/docker/controller.py":
        raise ValueError("Creator controller import came from another source tree")

    def factory(
        image: str,
        snapshot_name: str = "sci_bench",
        docker_binary: str = "docker",
        startup_timeout: float = 240,
        run_args: list[str] | None = None,
    ) -> ManagedDesktop:
        if image != receipt["image_id"] or docker_binary != "docker" or run_args:
            raise ValueError("Unreviewed desktop image/runtime override")
        return ManagedDesktop(
            image,
            controller_factory=module.DockerController,
            snapshot_name=snapshot_name,
            startup_timeout=startup_timeout,
            receipt_dir=receipt_dir,
        )

    return factory
