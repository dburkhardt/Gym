# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted isolated controller entrypoint; scientific solver lives in owned Docker."""

import argparse
import asyncio
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path

from benchmarks.bioagentbench.execution import run_native
from nemo_gym.dataset_staging import stage_file
from nemo_gym.sandbox.guarded_chat_transport import publish


def validate_identity(packet: dict, config: dict) -> dict:
    """Validate selected sources and actual native imports before any dispatch."""
    pins = json.loads(Path(__file__).with_name("source-sha256.json").read_text())
    root = Path(packet["creator_root"]).resolve()
    for relative, expected in pins["creator"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Creator source drift: " + relative)
    modules = {}
    metadata = Path(config["taskset"]["metadata_path"])
    if hashlib.sha256(metadata.read_bytes()).hexdigest() != pins["metadata_sha256"]:
        raise ValueError("Frozen metadata changed")
    for name in (
        "benchmarks.bioagentbench.execution",
        "benchmarks.bioagentbench.runtime",
        "benchmarks.bioagentbench.transport",
        "nemo_gym.sandbox.api",
        "nemo_gym.sandbox.lifecycle",
        "nemo_gym.sandbox.providers.docker.provider",
        "nemo_gym.sandbox.providers.local.provider",
        "nemo_gym.server_utils",
    ):
        module = importlib.import_module(name)
        actual = str(Path(module.__file__).resolve())
        if actual not in packet["selected_sources"]:
            raise ValueError("Unbound actual Gym import: " + name)
        modules[name] = module
    if str(Path(__file__).resolve()) not in packet["selected_sources"]:
        raise ValueError("Worker imported outside selected source")
    modules["bioagent_worker"] = sys.modules[__name__]
    for path, expected in packet["selected_sources"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("Selected Gym source changed")
    if (
        os.path.abspath(sys.executable) != packet["parent_interpreter"]["path"]
        or hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest()
        != packet["parent_interpreter"]["sha256"]
    ):
        raise ValueError("Native interpreter changed")
    manifest_path = Path(packet["asset_manifest"])
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != packet["asset_manifest_sha256"]:
        raise ValueError("Asset manifest changed")
    asset_root = Path(config["taskset"]["data_root"]) / packet["task_id"]
    if asset_root.is_symlink() or any(parent.is_symlink() for parent in asset_root.parents):
        raise ValueError("Asset root traverses a symlink")
    assets = json.loads(manifest_path.read_text())["tasks"][packet["task_id"]]["files"]
    actual = {}
    for category in ("data", "reference", "results"):
        directory = asset_root / category
        if directory.is_symlink():
            raise ValueError("Asset category is a symlink")
        for file in directory.rglob("*"):
            if file.is_symlink():
                raise ValueError("Asset tree contains a symlink")
            if file.is_file():
                actual[str(file.relative_to(asset_root))] = hashlib.sha256(file.read_bytes()).hexdigest()
    if actual != assets or not any(name.startswith("data/") for name in actual):
        raise ValueError("Complete selected public/reference/truth asset packet differs from its manifest")
    return {
        "python": sys.version,
        "interpreter": os.path.abspath(sys.executable),
        "binary_path": str(Path(sys.executable).resolve()),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "distributions": sorted((d.metadata.get("Name", ""), d.version) for d in importlib.metadata.distributions()),
        "imports": {
            name: {"path": module.__file__, "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
            for name, module in modules.items()
        },
        "selected_sources": packet["selected_sources"],
    }


def stage_owned_inputs(packet: dict, config: dict, directory: Path) -> dict:
    """Copy the exact input packet to owned writable state using shared publication."""
    source = Path(config["taskset"]["data_root"]) / packet["task_id"]
    manifest_bytes = Path(packet["asset_manifest"]).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != packet["asset_manifest_sha256"]:
        raise ValueError("Asset manifest changed before staging")
    files = json.loads(manifest_bytes)["tasks"][packet["task_id"]]["files"]
    from pathlib import PurePosixPath

    for name in files:
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) < 2
            or relative.parts[0] not in ("data", "reference", "results")
        ):
            raise ValueError("Unsafe asset manifest path")
        original = source / name
        if original.is_symlink() or any(p.is_symlink() for p in original.parents) or not original.is_file():
            raise ValueError("Asset source is not a regular file under the selected tree")
    required = sum((source / name).stat().st_size for name in files)
    if shutil.disk_usage(directory).free - required < 8 * 1024**3:
        raise RuntimeError("Full native input copy would cross the 8 GiB disk reserve")
    target = directory / "owned-inputs" / packet["task_id"]
    target.mkdir(parents=True, exist_ok=False)
    for name, digest in files.items():
        destination = target / name
        stage_file(source / name, destination, expected_sha256=digest)
        # This fresh task directory has one writer; do not expose publication locks as scientific inputs.
        destination.with_name(f".{destination.name}.stage.lock").unlink(missing_ok=True)
    selected = copy.deepcopy(config)
    selected["taskset"]["data_root"] = str(target.parent)
    publish(directory / "owned-inputs.json", {"task_id": packet["task_id"], "bytes": required, "files": files})
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check actual runtime/source/assets without provisioning or inference",
    )
    args = parser.parse_args()
    packet = json.loads(args.packet.read_text())
    from benchmarks.bioagentbench.source_stage import snapshot_creator

    packet["creator_root"] = str(snapshot_creator(Path(packet["creator_root"]), args.packet.parent / "creator-source"))
    config = packet["native_config"]
    identity = validate_identity(packet, config)
    # Import qualification runs under the creator lock, never this Gym environment.
    import subprocess

    child_packet = args.packet.parent / "creator-validation-packet.json"
    publish(child_packet, packet)
    validation = subprocess.run(
        [
            packet["interpreter"]["path"],
            "-m",
            "benchmarks.bioagentbench.creator_worker",
            "--packet",
            str(child_packet),
            "--validate-only",
        ],
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(args.packet.parent),
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        },
        capture_output=True,
        timeout=60,
    )
    (args.packet.parent / "creator-validation.log").write_bytes(validation.stdout + validation.stderr)
    if validation.returncode:
        raise RuntimeError("Exact creator runtime import qualification failed before execution")
    identity["creator_runtime"] = json.loads((args.packet.parent / "creator-imports.json").read_text())
    packet["digest"] = hashlib.sha256(
        json.dumps({"selected_packet_digest": packet["digest"], "actual_runtime": identity}, sort_keys=True).encode()
    ).hexdigest()
    identity["dispatch_config_sha256"] = packet["digest"]
    publish(args.packet.parent / "actual-imports.json", identity)
    if args.validate_only:
        return
    config = stage_owned_inputs(packet, config, args.packet.parent)

    async def execute() -> list[dict]:
        from nemo_gym.server_utils import GlobalAIOHTTPAsyncClientConfig, set_global_aiohttp_client

        client = set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
        try:
            return await run_native(config=config, packet=packet, output=args.packet.parent / "execution")
        finally:
            await client.close()

    traces = asyncio.run(execute())
    publish(args.packet.parent / "result.json", {"traces": traces})


if __name__ == "__main__":
    main()
