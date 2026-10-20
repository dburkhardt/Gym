# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One owned CPU oracle sandbox per verification; no inference or request retries."""

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from aiohttp import ClientError, ClientTimeout
from pydantic import BaseModel, Field

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.server_utils import request


PINNED_IMAGE = "sha256:2497011848c8b0649f967a61d7061fe9c6d1f121f30d4006c66c50c7545f4449"


class ManagedOracleConfig(BaseModel):
    source_dir: Path
    source_sha256: str
    data_dir: Path
    data_sha256: str
    artifact_dir: Path
    startup_timeout_seconds: float = Field(default=180, gt=0)
    lifetime_seconds: int = Field(default=900, gt=0)


def _publish(path: Path, value: dict) -> None:
    """Durably publish a complete recovery receipt before continuing."""
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, indent=2).encode())
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _data_hash(root: Path) -> str:
    manifest = "".join(
        f"{p.relative_to(root)} {_file_hash(p)}\n"
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.name.startswith("._")
    )
    if not manifest:
        raise ValueError("Empty native oracle dataset")
    return hashlib.sha256(manifest.encode()).hexdigest()


async def _settle(task: asyncio.Task):
    """Let bounded Docker creation/cleanup finish even under repeated cancellation."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


@asynccontextmanager
async def managed_oracle(config: ManagedOracleConfig):
    """Yield a dynamic endpoint; preserve receipts and kill all children on exit.

    The sandbox TTL also bounds orphan lifetime after a controller process crash.
    Stop failures propagate, with the serialized owned handle retained for recovery.
    """
    for path in (config.source_dir, config.data_dir):
        if not path.is_dir():
            raise ValueError(f"Missing pinned oracle assets: {path}")
    source_files = sorted(p for p in config.source_dir.rglob("*.py") if not p.name.startswith("._"))
    source_manifest = "".join(
        f"{p.relative_to(config.source_dir)} {hashlib.sha256(p.read_bytes()).hexdigest()}\n" for p in source_files
    )
    source_hash = hashlib.sha256(source_manifest.encode()).hexdigest()
    if not source_files or source_hash != config.source_sha256:
        raise ValueError("Pinned native oracle Python source hash mismatch")
    data_hash = await asyncio.to_thread(_data_hash, config.data_dir)
    if data_hash != config.data_sha256:
        raise ValueError("Pinned native oracle dataset hash mismatch")
    identity = uuid4().hex
    artifacts = config.artifact_dir.resolve() / identity
    artifacts.mkdir(parents=True, exist_ok=False)
    (artifacts / "scratch").mkdir()
    service = Path(__file__).with_name("oracle_service.py")
    shutil.copyfile(service, artifacts / "serve.py")
    proof = {
        "source_sha256": source_hash,
        "data_sha256": data_hash,
        "image": PINNED_IMAGE,
        "service_sha256": hashlib.sha256(service.read_bytes()).hexdigest(),
        "variant": "CPU PyScreener/Vina1.1.2",
        "run": identity,
        "cleanup_confirmed": False,
    }
    receipt = artifacts / "lifecycle.json"
    _publish(receipt, proof)
    spec = SandboxSpec(
        image=PINNED_IMAGE,
        ttl_s=config.lifetime_seconds,
        resources=SandboxResources(cpu=2, memory_mib=3072),
        ports=(37000,),
        metadata={"workload": "molrgen-oracle", "run": identity},
        provider_options={
            "volumes": [
                f"{artifacts}:/control",
                f"{artifacts}/scratch:/tmp/pyscreener",
                f"{config.source_dir.resolve()}:/opt/molrgen:ro",
                f"{config.data_dir.resolve()}:/data:ro",
            ],
            "run_args": ["--shm-size", "256m"],
        },
    )
    sandbox = AsyncSandbox({"docker": {}}, spec)
    start = asyncio.create_task(sandbox.start())
    try:
        await asyncio.shield(start)
        proof["sandbox"] = await sandbox.serialize()
        _publish(receipt, proof)
        launch = await sandbox.exec("python /control/serve.py > /control/service.log 2>&1 &", timeout_s=15)
        if launch.return_code:
            raise RuntimeError(f"Oracle launch failed: {launch.stderr}")
        endpoint = (await sandbox.endpoint(37000)).endpoint
        proof["endpoint"] = endpoint
        _publish(receipt, proof)
        deadline = time.monotonic() + config.startup_timeout_seconds
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Native oracle did not become ready")
            try:
                received = await request(
                    "GET", endpoint, timeout=ClientTimeout(total=2), _max_num_tries=1, _resend_after_disconnect=False
                )
                async with received:
                    received.raise_for_status()
                    if (await received.json(content_type=None))["status"] == "ready":
                        break
            except (ClientError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Native oracle did not become ready") from None
                await asyncio.sleep(0.5)
        yield endpoint, artifacts
    finally:
        # Creation is shielded: do not leave a late-created container behind.
        try:
            await _settle(start)
        finally:
            await _settle(asyncio.create_task(sandbox.stop()))
            proof["cleanup_confirmed"] = True
            _publish(receipt, proof)
