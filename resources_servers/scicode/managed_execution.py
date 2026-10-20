# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated creator-compatible execution with durable owned sandbox receipts."""

import asyncio
import hashlib
import json
import os
import re
import shlex
from pathlib import Path
from uuid import uuid4

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec


def _publish(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


async def _settle(task: asyncio.Task):
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    value = task.result()
    if cancelled:
        raise asyncio.CancelledError()
    return value


async def execute_substep(
    *, program: str, data_path: Path, image: str, artifact_dir: Path, timeout_secs: float, cpu: float, memory_mib: int
) -> dict:
    """Execute generated code only inside an attributed network-disabled sandbox."""
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("SciCode sandbox image must be an immutable sha256 image ID")
    identity = uuid4().hex
    directory = artifact_dir.resolve() / identity
    directory.mkdir(parents=True, exist_ok=False)
    directory.chmod(0o755)
    source = directory / "program.py"
    source.write_text(program)
    source.chmod(0o444)
    wrapper = Path(__file__).with_name("sandbox") / "execute.py"
    proof = {
        "run": identity,
        "image": image,
        "program_sha256": hashlib.sha256(program.encode()).hexdigest(),
        "wrapper_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        "cleanup_confirmed": False,
    }
    receipt = directory / "lifecycle.json"
    _publish(receipt, proof)
    sandbox = AsyncSandbox(
        {
            "docker": {
                "create": {
                    "network": "none",
                    "read_only": True,
                    "cap_drop": ["ALL"],
                    "security_opt": ["no-new-privileges"],
                    "pids_limit": 128,
                }
            }
        },
        SandboxSpec(
            image=image,
            ttl_s=timeout_secs + 90,
            resources=SandboxResources(cpu=cpu, memory_mib=memory_mib),
            metadata={"workload": "scicode-verifier", "run": identity},
            workdir="/tmp",
            provider_options={
                "volumes": [
                    f"{source}:/input/program.py:ro",
                    f"{data_path.resolve()}:/data/test_data.h5:ro",
                    f"{wrapper.resolve()}:/control/execute.py:ro",
                ],
                "run_args": ["--user", "65534:65534", "--tmpfs", "/tmp:rw,nosuid,size=256m"],
            },
        ),
    )
    start = asyncio.create_task(sandbox.start())
    original_error = None
    try:
        await asyncio.shield(start)
        proof["sandbox"] = await sandbox.serialize()
        _publish(receipt, proof)
        preflight = (
            "import sys, scipy, h5py; from scipy.integrate import simps; "
            "assert sys.version_info[:2] == (3, 12); assert scipy.__version__ == '1.13.1'; "
            "handle = h5py.File('/data/test_data.h5', 'r'); handle.close()"
        )
        ready = await sandbox.exec("python -c " + shlex.quote(preflight), timeout_s=15)
        if ready.return_code != 0:
            raise RuntimeError("SciCode scientific runtime/data preflight failed: " + ready.stderr[-2000:])
        result = await sandbox.exec(f"python /control/execute.py {float(timeout_secs)}", timeout_s=timeout_secs + 15)
        if result.return_code != 0:
            raise RuntimeError(f"SciCode sandbox execution failed: {result.stderr[-2000:]}")
        outcome = json.loads(result.stdout)
        outcome["receipt_path"] = str(receipt)
        _publish(directory / "result.json", outcome)
        if outcome["return_code"] < 0 and not outcome["timed_out"]:
            raise RuntimeError(
                f"SciCode generated process terminated by signal {-outcome['return_code']}; "
                "execution validity unresolved"
            )
        return outcome
    except BaseException as error:
        original_error = error
        proof["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        try:
            try:
                await _settle(start)
                proof["sandbox"] = await sandbox.serialize()
                _publish(receipt, proof)
            finally:
                await _settle(asyncio.create_task(sandbox.stop()))
            proof["cleanup_confirmed"] = True
            _publish(receipt, proof)
        except BaseException as error:
            proof["cleanup_error"] = {"type": type(error).__name__, "message": str(error)}
            _publish(receipt, proof)
            if original_error is None:
                raise
