# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted creator subprocess; solver tools remain in the owned offline viewer."""

import asyncio
import hashlib
import importlib
import inspect
import json
import os
import shutil
import sys
from pathlib import Path

import httpx

from benchmarks.scivisagentbench.evaluate import evaluate
from benchmarks.scivisagentbench.native import run_native
from benchmarks.scivisagentbench.transport import MODEL
from nemo_gym.guarded_chat_transport import GuardedChatTransport, publish


def runtime_identity(output: Path, expected: dict) -> None:
    paths = [
        Path(importlib.import_module("benchmarks.scivisagentbench." + n).__file__).resolve()
        for n in ("native", "transport", "judge", "evaluate")
    ]
    paths.append(Path(__file__).resolve())
    paths.append(Path(inspect.getfile(GuardedChatTransport)).resolve())
    actual = {
        "source": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "interpreter": {
            "path": str(Path(sys.executable).absolute()),
            "resolved_path": str(Path(sys.executable).resolve()),
            "sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        },
        "python_version": sys.version,
    }
    publish(output / "runtime-imports.json", actual)
    if actual["interpreter"] != expected["interpreter"] or any(
        expected["source"].get(p) != sha for p, sha in actual["source"].items()
    ):
        raise ValueError("Creator child import/runtime identity mismatch")


def stage_judge_data(workspace: Path, destination: Path, reference: Path, expected: dict) -> None:
    if any(p.is_symlink() for p in workspace.rglob("*")):
        raise ValueError("Solver artifacts contain symlinks; do not follow host paths")
    shutil.copytree(workspace, destination)
    if reference.is_symlink():
        raise ValueError("Pinned judge reference root is a symlink")
    reference = reference.resolve()
    for relative, sha in expected.items():
        path = reference / relative
        if (
            not relative.startswith("GS/")
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not path.resolve().is_relative_to(reference)
            or any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(reference))
            or hashlib.sha256(path.read_bytes()).hexdigest() != sha
        ):
            raise ValueError("Pinned judge reference mismatch")
        target = destination / relative
        if target.exists():
            raise ValueError("Solver output collides with judge reference")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


async def main(output: Path) -> None:
    identity = json.loads((output / "identity.json").read_text())
    runtime_identity(output, identity)
    payload = json.loads((output / "worker-input.json").read_text())
    task = payload["task"]
    transport = GuardedChatTransport(
        socket_path=os.environ["SCIVIS_SOCKET"],
        token=os.environ["SCIVIS_TOKEN"],
        lineage=os.environ["SCIVIS_LINEAGE"],
        digest=os.environ["SCIVIS_DIGEST"],
        namespace="scivis",
        model=MODEL,
        output=output / "provider-calls",
    )
    # Shared transport persists the raw terminal body before handing it to native adapters.
    stopped = False
    request_timeout_s = identity.get("request_timeout_s", 600)
    async with httpx.AsyncClient(transport=transport, timeout=request_timeout_s, follow_redirects=False) as client:

        async def sender(body):
            if stopped:
                raise RuntimeError("Native phase stopped; no further dispatch")
            response = await client.post("http://gym-internal/chat/completions", json=body)
            response.raise_for_status()
            return response.json()

        if "saved_solver" in payload:
            solver = payload["saved_solver"]
        else:
            async with asyncio.timeout(identity["timeout_s"]):
                solver = await run_native(
                    Path(payload["source_root"]),
                    payload["native_config"],
                    task,
                    output / "solver",
                    sender,
                    identity["max_tokens"],
                )
        evaluation = {}
        if solver.get("success"):
            judge_started = asyncio.get_running_loop().time()
            if "saved_solver" not in payload:
                exporter = await asyncio.create_subprocess_exec(
                    payload["artifact_export_python"], "-m", "benchmarks.scivisagentbench.artifacts", str(output),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(exporter.communicate(), 120)
                except BaseException:
                    if exporter.returncode is None:
                        exporter.kill()
                    await exporter.communicate()
                    raise
                (output / "artifact-export.stdout").write_bytes(stdout)
                (output / "artifact-export.stderr").write_bytes(stderr)
                if exporter.returncode:
                    raise RuntimeError("Owned artifact export failed")
            stage_judge_data(
                output / "exported-artifacts",
                Path(task["judge_data_dir"]),
                Path(payload["reference_root"]),
                payload["reference_sha256"],
            )
            loop = asyncio.get_running_loop()

            def sync_sender(body):
                return asyncio.run_coroutine_threadsafe(sender(body), loop).result(timeout=request_timeout_s + 10)

            # Native evaluator stays synchronous internally; only its execution thread changes.
            judging = asyncio.create_task(
                asyncio.to_thread(
                    lambda: asyncio.run(
                        evaluate(
                            Path(payload["source_root"]),
                            task["source_case"],
                            Path(task["judge_case_dir"]),
                            Path(task["judge_data_dir"]),
                            task["agent_mode"],
                            sync_sender,
                            output / "evaluation",
                        )
                    )
                )
            )
            try:
                evaluation = await asyncio.wait_for(
                    asyncio.shield(judging),
                    max(0, identity["timeout_s"] - (loop.time() - judge_started)),
                )
            except BaseException:
                # Stop fresh dispatch, then settle an already accepted response/thread.
                stopped = True
                transport.failed = True
                while not judging.done():
                    try:
                        await asyncio.shield(judging)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if judging.done() and not judging.cancelled():
                    judging.exception()
                raise
        publish(output / "worker-result.json", {"solver": solver, "evaluation": evaluation})


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]).resolve()))
