# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public-only generator using the shared Docker lifecycle and guarded Unix bridge."""

import hashlib
import json
import math
import os
import re
import secrets
import time
from pathlib import Path

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.lifecycle import _owned_sandbox
from nemo_gym.sandbox.providers.docker import DockerCreateConfig, DockerProvider
from nemo_gym.sandbox.providers.local.provider import LocalProvider
from nemo_gym.sandbox.scoring_workspace import scoring_workspace
from responses_api_agents.metasyn_native.identity import creator_identity


def export_public(creator: Path, prompt: Path, destination: Path) -> dict[str, str]:
    creator_identity(creator)
    if prompt.is_symlink() or not prompt.is_file():
        raise ValueError("Exact public prompt file required")
    destination.mkdir(exist_ok=False)
    for name, path in (("rag.py", creator / "rag.py"), ("prompt.txt", prompt)):
        (destination / name).write_bytes(path.read_bytes())
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir()}


def generator_spec(
    image: str, public: Path, control: Path, output: Path, socket: Path, *, remaining: float, lineage: str
) -> SandboxSpec:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None or not math.isfinite(remaining) or remaining <= 0:
        raise ValueError("Immutable qualified generator image and remaining phase time required")
    if set(public.iterdir()) != {public / "rag.py", public / "prompt.txt"}:
        raise ValueError("Public generator mount allowlist changed")
    if set(control.iterdir()) != {control / "public_generator.py", control / "guarded_chat_transport.py"}:
        raise ValueError("Minimal generator control mount required")
    if set(socket.parent.iterdir()) != {socket}:
        raise ValueError("Private socket mount contains other files")
    for folder in (public, control, output, socket.parent):
        if folder.is_symlink() or not folder.is_dir() or any(p.is_symlink() for p in folder.iterdir()):
            raise ValueError("Generator mount symlink forbidden")
    return SandboxSpec(
        image=image,
        workdir="/workspace",
        ttl_s=remaining + 180,
        resources=SandboxResources(cpu=1, memory_mib=2048),
        metadata={"workload": "metasyn-public-generator", "run": lineage},
        provider_options={
            "volumes": [
                f"{public.resolve()}:/public:ro",
                f"{control.resolve()}:/control:ro",
                f"{output.resolve()}:/workspace",
                f"{socket.parent.resolve()}:/bridge:ro",
            ],
            "run_args": ["--user", f"{os.getuid()}:{os.getgid()}", "--tmpfs", "/tmp:rw,nosuid,nodev,size=128m"],
        },
    )


async def run_generator(
    *,
    image: str,
    creator: Path,
    prompt: Path,
    out: Path,
    endpoint: str,
    lineage: str,
    digest: str,
    deadline: float,
    expected_prompt_sha256: str,
) -> Path:
    from nemo_gym.sandbox.model_bridge import bridge

    public, control, output = (out / name for name in ("generator-public", "generator-control", "generator-output"))
    hashes = export_public(creator, prompt, public)
    if hashes["prompt.txt"] != expected_prompt_sha256:
        raise ValueError("Generator prompt differs from parent scientific identity")
    control.mkdir()
    output.mkdir()
    import nemo_gym.guarded_chat_transport as shared

    for name, path in (
        ("public_generator.py", Path(__file__).with_name("public_generator.py")),
        ("guarded_chat_transport.py", Path(shared.__file__)),
    ):
        (control / name).write_bytes(path.read_bytes())
    control_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in control.iterdir()}
    binding = {
        "image": image,
        "public": hashes,
        "control": control_hashes,
        "parent_config": digest,
        "lineage": lineage,
        "deadline": deadline,
    }
    (out / "generator-binding.json").write_text(json.dumps(binding, indent=2))
    phase_digest = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    async with scoring_workspace(LocalProvider(concurrency=1)) as handle:
        socket = handle.raw["workspace"] / "model.sock"
        token = secrets.token_urlsafe(32)
        async with bridge(
            socket,
            endpoint=endpoint,
            token=token,
            lineage=lineage + ":generator",
            digest=phase_digest,
            namespace="metasyn",
            expected_model="nvidia/nvidia/nemotron-3.5-super-vl",
        ):
            spec = generator_spec(
                image, public, control, output, socket, remaining=deadline - time.time(), lineage=lineage
            )
            provider = DockerProvider(
                create=DockerCreateConfig(
                    network="none", read_only=True, cap_drop=["ALL"], security_opt=["no-new-privileges"]
                )
            )
            box = AsyncSandbox(provider, spec)
            async with _owned_sandbox(box, out / "generator-lifecycle.json"):
                (out / "generator-sandbox.json").write_text(json.dumps(await box.serialize(), indent=2))
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("MetaSyn combined deadline exhausted during container startup")
                result = await box.exec(
                    "python -B /control/public_generator.py",
                    timeout_s=remaining,
                    env={
                        "METASYN_BRIDGE_TOKEN": token,
                        "METASYN_BRIDGE_LINEAGE": lineage + ":generator",
                        "METASYN_BRIDGE_DIGEST": phase_digest,
                        "METASYN_PUBLIC_HASHES": json.dumps(hashes),
                    },
                )
                (out / "generator-execution.json").write_text(
                    json.dumps(
                        {
                            "return_code": result.return_code,
                            "error_type": result.error_type,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                        },
                        indent=2,
                    )
                )
            if result.return_code or result.error_type:
                raise RuntimeError("Isolated generator failed; no verifier execution or automatic retry")
    artifact = output / "generation.json"
    if artifact.is_symlink() or not artifact.is_file():
        raise ValueError("Missing regular generator artifact")
    (out / "generation-sha256.json").write_text(
        json.dumps({"sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()})
    )
    return artifact
