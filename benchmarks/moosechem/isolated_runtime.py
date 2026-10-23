# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact public mounts and owned network-disabled MOOSE execution."""

import hashlib
import json
import math
import os
import shlex
import stat
import time
from pathlib import Path

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.lifecycle import _owned_sandbox
from nemo_gym.sandbox.providers.docker import DockerCreateConfig, DockerProvider


MODULES = (
    "native.py",
    "guarded.py",
    "isolated.py",
    "isolated_worker.py",
    "qualify_replay.py",
    "qualify_settled_tail.py",
    "source-sha256.json",
)


def exact_tree(root: Path, expected: dict[str, str]) -> None:
    if root.is_symlink() or not root.is_dir() or not expected:
        raise ValueError("Exact owned tree required")
    found = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Mount symlinks forbidden")
        if path.is_file():
            found[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if found != expected:
        raise ValueError("Mount contents changed")


def stage_modules(destination: Path) -> dict[str, str]:
    """Publish only executable source needed by the child, never repository data."""
    destination.mkdir(parents=True, exist_ok=False)
    module = destination / "benchmarks/moosechem"
    module.mkdir(parents=True)
    for name in MODULES:
        (module / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    return {
        str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in destination.rglob("*")
        if p.is_file()
    }


def phase_spec(
    *,
    image: str,
    source: Path,
    source_hashes: dict,
    repo: Path,
    repo_hashes: dict,
    output: Path,
    socket: Path,
    timeout: float,
    lineage: str,
    phase: str,
    evidence: Path | None = None,
    evidence_hashes: dict | None = None,
    public_manifest_sha256: str | None = None,
    submission: Path | None = None,
    submission_hashes: dict | None = None,
) -> SandboxSpec:
    if not image.startswith("sha256:") or len(image) != 71 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Immutable runtime and positive remaining timeout required")
    if phase not in ("solver", "verifier"):
        raise ValueError("Unknown native phase")
    if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
        raise ValueError("Fresh empty owned phase output required")
    if socket.is_symlink() or socket.name != "model.sock" or not stat.S_ISSOCK(socket.stat().st_mode):
        raise ValueError("Expected owned Unix socket required")
    if set(socket.parent.iterdir()) != {socket} or socket.parent.is_symlink():
        raise ValueError("Bridge workspace contains unexpected files")
    exact_tree(source, source_hashes)
    exact_tree(repo, repo_hashes)
    if phase == "solver":
        from benchmarks.moosechem.isolated import validate_public

        validate_public(repo, repo / "task.json", expected_manifest_sha256=public_manifest_sha256)
    volumes = [
        f"{source.resolve()}:/control:ro",
        f"{repo.resolve()}:/repo:ro",
        f"{output.resolve()}:/workspace",
        f"{socket.parent.resolve()}:/bridge:ro",
    ]
    if evidence is not None:
        exact_tree(evidence, evidence_hashes or {})
        if set(evidence_hashes or {}) != {"saved.jsonl", "gateway.jsonl", "settlement.json"}:
            raise ValueError("Only explicit settlement evidence may enter solver")
        volumes.append(f"{evidence.resolve()}:/settlement:ro")
    if submission is not None:
        if phase != "verifier":
            raise ValueError("Solver cannot mount a verifier submission")
        exact_tree(submission, submission_hashes or {})
        volumes.append(f"{submission.resolve()}:/submission:ro")
    elif phase == "verifier":
        raise ValueError("Verifier needs an exact saved solver submission")
    return SandboxSpec(
        image=image,
        workdir="/workspace",
        ttl_s=timeout + 180,
        resources=SandboxResources(cpu=1, memory_mib=4096),
        metadata={"workload": "moosechem-isolated", "run": lineage, "phase": phase},
        provider_options={
            "volumes": volumes,
            "run_args": ["--user", f"{os.getuid()}:{os.getgid()}", "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m"],
        },
    )


async def execute_phase(spec: SandboxSpec, output: Path, args: list[str], env: dict, *, deadline: float) -> dict:
    """Shared lifecycle owns the container; preserve terminal bytes before checking status."""
    provider = DockerProvider(
        create=DockerCreateConfig(network="none", read_only=True, cap_drop=["ALL"], security_opt=["no-new-privileges"])
    )
    box = AsyncSandbox(provider, spec)
    async with _owned_sandbox(box, output / "lifecycle.json"):
        (output / "sandbox-descriptor.json").write_text(json.dumps(await box.serialize(), indent=2))
        remaining = deadline - time.time()
        if not math.isfinite(remaining) or remaining <= 0:
            receipt = {"return_code": None, "error_type": "deadline_before_exec"}
            (output / "execution.json").write_text(json.dumps(receipt, indent=2))
            return receipt
        result = await box.exec(
            shlex.join(["python", "-m", "benchmarks.moosechem.isolated_worker", *args]),
            timeout_s=remaining,
            env={**env, "PYTHONPATH": "/control"},
        )
        (output / "stdout.log").write_text(result.stdout or "")
        (output / "stderr.log").write_text(result.stderr or "")
        receipt = {"return_code": result.return_code, "error_type": result.error_type}
        (output / "execution.json").write_text(json.dumps(receipt, indent=2))
    return receipt
