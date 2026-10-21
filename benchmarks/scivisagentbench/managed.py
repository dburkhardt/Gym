# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned viewer/MCP sandbox and trusted creator process on shared Gym lifecycle."""

import asyncio
import hashlib
import inspect
import json
import secrets
import shlex
import uuid
from pathlib import Path

from benchmarks.scivisagentbench.native import validate_source
from benchmarks.scivisagentbench.transport import MODEL
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.lifecycle import _owned_sandbox, _publish
from nemo_gym.sandbox.model_bridge import bridge
from nemo_gym.sandbox.providers.local.provider import LocalProvider
from nemo_gym.sandbox.scoring_workspace import scoring_workspace


def checked_tree(root: Path, hashes: dict) -> None:
    """Require an exact, reference-free solver/runtime tree, not a broad host mount."""
    if not root.is_dir() or root.is_symlink() or not hashes:
        raise ValueError("Qualified tree and nonempty file manifest required")
    files = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Runtime mounts cannot contain symlinks")
        if path.is_file():
            relative = str(path.relative_to(root))
            if any(x.lower() in ("gs", "references", ".secrets", ".git") for x in path.relative_to(root).parts):
                raise ValueError("Reference/secret directory in solver mount")
            files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if files != hashes:
        raise ValueError("Runtime mount file/hash mismatch")


def viewer_spec(packet: dict, workspace: Path, lineage: str, timeout: float) -> SandboxSpec:
    if not packet["image"].startswith("sha256:") or len(packet["image"]) != 71:
        raise ValueError("Immutable qualified image ID required")
    if any(
        any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")) for key in packet.get("env", {})
    ):
        raise ValueError("Credentials must not enter the solver viewer environment")
    mounts = []
    for mount in packet["mounts"]:
        target = mount["target"]
        if target not in ("/source", "/workspace/data") and not target.startswith("/runtime/"):
            raise ValueError("Unexpected solver mount target")
        source = Path(mount["source"])
        checked_tree(source, mount["sha256"])
        source = source.resolve()
        mounts.append(f"{source}:{target}:ro")
    targets = [m["target"] for m in packet["mounts"]]
    if len(set(targets)) != len(targets) or "/source" not in targets or "/workspace/data" not in targets:
        raise ValueError("Unique source/data mounts required")
    # The prepared source tree must contain only the audited solver implementation.
    controls = Path(__file__).resolve().parent
    mounts += [
        f"{workspace.resolve()}:/workspace",
        f"{controls / 'viewer.py'}:/control/viewer.py:ro",
        f"{controls / 'mcp_bootstrap.py'}:/control/mcp_bootstrap.py:ro",
    ]
    return SandboxSpec(
        image=packet["image"],
        ttl_s=timeout + 180,
        workdir="/workspace",
        resources=SandboxResources(cpu=2, memory_mib=16384),
        metadata={"workload": "gym-scivis", "run": lineage},
        env=packet.get("env", {}),
        provider_options={
            "volumes": mounts,
            "run_args": [
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=512m",
                "--tmpfs",
                "/home/control:rw,nosuid,nodev,size=32m",
                "--ulimit",
                "core=0",
            ],
        },
    )


async def execute(config, task: dict, output: Path, endpoint: str) -> tuple[dict, dict]:
    """A fresh native attempt; no reconnection or replay is inferred from artifacts."""
    runtime_cases = json.loads(Path(config.runtime_packet).read_text())["cases"]
    if set(runtime_cases) != {f"case_{i}" for i in range(1, 12)}:
        raise ValueError("Runtime packet must retain all eleven frozen cases")
    packet = runtime_cases[task["case_name"]]
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    workspace.mkdir()
    lineage = uuid.uuid4().hex
    source = Path(__file__).resolve().parents[2]
    modules = [
        Path(__file__).with_name(n)
        for n in (
            "managed.py",
            "worker.py",
            "native.py",
            "transport.py",
            "judge.py",
            "evaluate.py",
            "viewer.py",
            "mcp_bootstrap.py",
        )
    ]
    modules.append(source / "responses_api_agents/scivis_native/app.py")
    modules.append(source / "nemo_gym/sandbox/guarded_chat_transport.py")
    imported = [
        Path(inspect.getfile(inspect.unwrap(obj))).resolve()
        for obj in (AsyncSandbox, LocalProvider, bridge, _owned_sandbox)
    ]
    if any(not p.is_relative_to(source) for p in imported):
        raise ValueError("Shared runtime imported outside selected Gym source")
    modules.extend(imported)
    interpreter = Path(config.native_python).resolve()
    native_config = json.loads(Path(config.native_config).read_text())
    identity = {
        "source": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in modules},
        "interpreter": {"path": str(interpreter), "sha256": hashlib.sha256(interpreter.read_bytes()).hexdigest()},
        "runtime_packet": packet,
        "native_config": native_config,
        "task": task,
        "model": MODEL,
        "max_tokens": config.max_tokens,
        "timeout_s": config.timeout_s,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    _publish(output / "identity.json", identity)
    creator = Path(config.source_root).resolve()
    creator_identity = validate_source(creator)
    _publish(output / "creator-source.json", creator_identity)
    for mount in packet["mounts"]:
        if mount["target"] == "/source":
            for relative, sha in mount["sha256"].items():
                if (
                    not relative.startswith("src/")
                    or ".." in Path(relative).parts
                    or hashlib.sha256((creator / relative).read_bytes()).hexdigest() != sha
                ):
                    raise ValueError("Solver source mount differs from pinned creator src tree")
    spec = viewer_spec(packet, workspace, lineage, 2 * config.timeout_s + 660)
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
        spec,
    )
    async with _owned_sandbox(sandbox, output / "viewer-lifecycle.json"):
        descriptor = await sandbox.serialize()
        # Actual creator MCP commands target only this owned container identity.
        container = descriptor["sandbox_id"]
        command = shlex.join(packet["viewer_command"]) + " > /workspace/viewer.log 2>&1 &"
        result = await sandbox.exec(command, timeout_s=30, preserve_background_services=True)
        if result.return_code:
            raise RuntimeError("Owned viewer startup failed")
        async with asyncio.timeout(90):
            while not (workspace / "viewer-ready.json").exists():
                await asyncio.sleep(0.1)
        native_config = json.loads(json.dumps(native_config))
        native_config["servers"] = [
            {
                "type": "stdio",
                "config": {
                    "command": "docker",
                    "args": [
                        "exec",
                        "-i",
                        "-e",
                        "TEST_CASE_NAME=" + task["case_name"],
                        container,
                        *packet["napari_mcp_command"],
                    ],
                },
            },
            {
                "type": "stdio",
                "config": {
                    "command": "docker",
                    "args": ["exec", "-i", container, *packet["filesystem_mcp_command"], "/workspace"],
                },
            },
        ]
        native_task = {
            **task,
            "judge_case_dir": str(output / "judge" / task["case_name"]),
            "judge_data_dir": str(output / "judge-data"),
        }
        required_refs = {a["gs-file"] for a in task["source_case"]["assert"] if "gs-file" in a}
        if set(packet["reference_sha256"]) != required_refs:
            raise ValueError("Exact frozen case reference files required")
        payload = {
            "task": native_task,
            "native_config": native_config,
            "source_root": config.source_root,
            "reference_root": packet["reference_root"],
            "reference_sha256": packet["reference_sha256"],
        }
        _publish(output / "worker-input.json", payload)
        provider = LocalProvider(concurrency=1, workspace_root="/tmp")
        async with scoring_workspace(provider) as handle:
            socket = handle.raw["workspace"] / "model.sock"
            token = secrets.token_urlsafe(32)
            async with bridge(
                socket,
                endpoint=endpoint,
                token=token,
                lineage=lineage,
                digest=digest,
                namespace="scivis",
                expected_model=MODEL,
            ):
                result = await provider.exec(
                    handle,
                    shlex.join([str(interpreter), "-m", "benchmarks.scivisagentbench.worker", str(output)]),
                    timeout_s=2 * config.timeout_s + 660,
                    env={
                        "PYTHONPATH": str(source),
                        "NEMO_GYM_EXTRA_ROOTS": str(source),
                        "SCIVIS_SOCKET": str(socket),
                        "SCIVIS_TOKEN": token,
                        "SCIVIS_LINEAGE": lineage,
                        "SCIVIS_DIGEST": digest,
                    },
                )
                (output / "worker.stdout").write_text(result.stdout or "")
                (output / "worker.stderr").write_text(result.stderr or "")
                if result.return_code or result.error_type:
                    raise RuntimeError("Native creator process failed; raw receipts retained")
    record = json.loads((output / "worker-result.json").read_text())
    return record["solver"], record["evaluation"]
