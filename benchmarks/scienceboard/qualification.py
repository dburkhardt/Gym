# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime qualification contract and exact owned desktop cleanup; no model calls."""

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from benchmarks.scienceboard.prepare import MANIFEST
from nemo_gym.sandbox.guarded_chat_transport import publish
from nemo_gym.sandbox.providers.docker.provider import DockerProvider


def implementation_identity(python: Path) -> dict:
    """Bind actual selected implementation and configured venv launcher."""
    source = Path(__file__).resolve().parents[2]
    files = [
        *MANIFEST.parent.glob("*.py"),
        MANIFEST,
        source / "responses_api_agents/scienceboard_native/app.py",
        *[
            source / "nemo_gym/sandbox" / name
            for name in (
                "api.py",
                "model_bridge.py",
                "guarded_chat_transport.py",
                "scoring_workspace.py",
                "providers/local/provider.py",
                "providers/docker/provider.py",
            )
        ],
    ]
    python = python.absolute()
    return dict(
        interpreter=str(python),
        interpreter_sha256=hashlib.sha256(python.read_bytes()).hexdigest(),
        modules={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    )


def runtime_identity() -> dict:
    """Record interpreter and installed distribution metadata; not an immutable OS image."""
    from urllib.parse import unquote, urlparse

    packages = []
    root = Path(__file__).resolve().parents[2]
    for dist in importlib.metadata.distributions():
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        if direct.get("dir_info", {}).get("editable"):
            origin = Path(unquote(urlparse(direct.get("url", "")).path)).resolve()
            if dist.metadata.get("Name", "").lower().replace("_", "-") != "nemo-gym" or origin != root:
                raise ValueError("External editable dependency cannot qualify ScienceBoard runtime")
        packages.append(
            dict(
                name=dist.metadata.get("Name"),
                version=dist.version,
                direct_url=direct,
                record_sha256=hashlib.sha256((dist.read_text("RECORD") or "").encode()).hexdigest(),
            )
        )
    loaded = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).is_file():
            path = Path(filename).resolve()
            loaded[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(
        executable=str(Path(sys.executable).absolute()),
        prefix=sys.prefix,
        python=platform.python_version(),
        loaded_modules=loaded,
        packages=sorted(packages, key=lambda x: (x["name"] or "", x["version"])),
    )


def validate_receipt(receipt: dict, *, implementation: dict, required: dict) -> dict:
    """Reject changed implementation, partial controls and missing environment identity."""
    if any(receipt.get(k) != v for k, v in required.items()) or receipt.get("implementation") != implementation:
        raise ValueError("Runtime receipt implementation/image/task/protocol differs")
    runtime = receipt.get("runtime")
    if (
        not isinstance(runtime, dict)
        or not runtime.get("packages")
        or runtime.get("executable") != implementation["interpreter"]
        or not runtime.get("loaded_modules")
    ):
        raise ValueError("Runtime receipt lacks exact interpreter/package identity")
    containers = receipt.get("cleanup", {}).get("containers", [])
    if (
        receipt.get("hard_death_signal") != "SIGKILL"
        or not receipt.get("cleanup", {}).get("all_removed")
        or not containers
        or any(
            not item.get("all_removed") or not item.get("ownership_labels", {}).get("nemo-gym.run")
            for item in containers
        )
    ):
        raise ValueError("Hard-death owned cleanup was not qualified")
    return runtime


def cleanup_desktops(directory: Path) -> dict:
    """Reconcile durable pre-create identities only after owned worker settlement."""
    import asyncio

    async def reconcile() -> dict:
        provider = DockerProvider()
        reports = []
        try:
            for path in sorted(directory.glob("*.intent.json")):
                intent = json.loads(path.read_text())
                report = await provider.reconcile_owned(intent["ownership_labels"])
                descriptor = path.with_name(path.name.replace(".intent.json", ".descriptor.json"))
                if not descriptor.exists() and not report["removed_ids"]:
                    report.update(all_removed=False, disposition="create_receipt_missing_no_container_observed")
                reports.append(report)
        finally:
            await provider.aclose()
        result = {"all_removed": all(x["all_removed"] for x in reports), "containers": reports}
        publish(directory.parent / "desktop-cleanup.json", result)
        if not result["all_removed"]:
            raise RuntimeError("Pre-create attempt has no settled receipt; cleanup remains unresolved")
        return result

    return asyncio.run(reconcile())


async def settle_cleanup(directory: Path) -> dict:
    """Finish exact owned cleanup despite repeated cancellation, preserving original error."""
    import asyncio

    original = sys.exception()
    interrupted = None
    task = asyncio.create_task(asyncio.to_thread(cleanup_desktops, directory))
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            interrupted = error
        except Exception:
            break
    try:
        result = task.result()
    except Exception as error:
        if original is not None:
            original.add_note(f"ScienceBoard owned cleanup also failed: {type(error).__name__}: {error}")
            return {"all_removed": False, "error": type(error).__name__}
        raise
    if interrupted is not None and original is None:
        raise interrupted
    return result


async def stop_worker(job, directory: Path) -> dict:
    """Settle the owned process before reconciliation even if settlement itself fails."""
    import asyncio

    try:
        job.cancel()
        while not job.done():
            try:
                await asyncio.shield(job)
            except asyncio.CancelledError:
                continue
        if not job.cancelled():
            job.result()
    finally:
        cleanup = await settle_cleanup(directory)
    return cleanup


async def qualify(packet: dict, python: Path, *, timeout_seconds: float = 600) -> dict:
    """Initialize one exact task, kill the owned controller, and reconnect/remove its desktop."""
    import asyncio
    import os
    import shlex
    from uuid import uuid4

    from benchmarks.scienceboard.admission import admit
    from benchmarks.scienceboard.guarded_model import PROTOCOL_VARIANT
    from nemo_gym.sandbox.providers.local.provider import LocalProvider
    from nemo_gym.sandbox.scoring_workspace import scoring_workspace

    if admit(packet["image"], packet["audit"], packet["resources"])["blockers"]:
        raise ValueError("Image evidence is not admitted")
    out = Path(packet["output"]).absolute()
    out.mkdir(parents=True, exist_ok=False)
    implementation = implementation_identity(python)
    packet = {
        **packet,
        "output": str(out),
        "implementation": implementation["modules"],
        "mode": "qualification",
        "socket": "/unused",
        "lineage": uuid4().hex,
        "digest": "0" * 64,
        "protocol": PROTOCOL_VARIANT,
    }
    publish(out / "packet.json", packet)
    source = Path(__file__).resolve().parents[2]
    provider = LocalProvider(concurrency=1, workspace_root="/tmp")
    async with scoring_workspace(provider) as handle:
        job = asyncio.create_task(
            provider.exec(
                handle,
                shlex.join(
                    [
                        str(python.absolute()),
                        "-m",
                        "benchmarks.scienceboard.worker",
                        "--packet",
                        str(out / "packet.json"),
                    ]
                ),
                timeout_s=timeout_seconds,
                terminate_background_children=True,
                env={
                    "PYTHONPATH": str(source) + os.pathsep + packet["source"],
                    "NEMO_GYM_EXTRA_ROOTS": str(source),
                    "SCIENCEBOARD_BRIDGE_TOKEN": "qualification-no-model-access",
                },
            )
        )
        initialized = False
        try:
            async with asyncio.timeout(timeout_seconds):
                while not (out / "initialized.json").exists():
                    if job.done():
                        result = await job
                        publish(
                            out / "worker-exit.json",
                            {"stdout": result.stdout, "stderr": result.stderr, "return_code": result.return_code},
                        )
                        raise RuntimeError("Creator initializer did not complete")
                    await asyncio.sleep(0.1)
                initialized = True
        finally:
            # LocalProvider cancellation explicitly SIGKILLs and reaps this process group.
            cleanup = await stop_worker(job, out / "desktops")
        if initialized:
            pid = json.loads((out / "initialized.json").read_text())["pid"]
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("Controller still exists after cancellation; SIGKILL settlement unproved")
        if not initialized or not cleanup.get("all_removed") or not cleanup["containers"]:
            raise RuntimeError("No initialized owned desktop was observed; qualification refused")
    manifest = json.loads(MANIFEST.read_text())
    receipt = dict(
        image_id=packet["image"]["Id"],
        source_revision=manifest["source_revision"],
        selection_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        protocol=PROTOCOL_VARIANT,
        task_id=packet["task_id"],
        desktop_initialized=True,
        controller_death_cleanup=True,
        implementation=implementation,
        runtime=json.loads((out / "runtime.json").read_text()),
        hard_death_signal="SIGKILL",
        cleanup=cleanup,
    )
    publish(out / "qualification.json", receipt)
    return receipt


def main() -> None:
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet", required=True, type=Path, help="Source/task/image/audit/resources/output JSON; no secrets"
    )
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.timeout <= 0:
        raise ValueError("Positive bounded timeout required")
    asyncio.run(qualify(json.loads(args.packet.read_text()), args.python, timeout_seconds=args.timeout))


if __name__ == "__main__":
    main()
