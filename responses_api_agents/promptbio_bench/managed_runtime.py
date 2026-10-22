# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native guarded PromptBio execution using shared sandbox/process lifecycles."""

import asyncio
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import sys
import tarfile
import time
import uuid
from pathlib import Path

import httpx
from pydantic import TypeAdapter

from nemo_gym.openai_utils import NeMoGymChatCompletionAssistantMessageParam

from nemo_gym.guarded_chat_transport import GuardedChatTransport
from nemo_gym.provider_dispatch import configuration_digest
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.lifecycle import _owned_sandbox, _publish
from nemo_gym.sandbox.model_bridge import bridge
from nemo_gym.sandbox.providers.base import SandboxResources, SandboxSpec
from nemo_gym.sandbox.providers.local.provider import LocalProvider
from nemo_gym.sandbox.scoring_workspace import scoring_workspace
from responses_api_agents.promptbio_bench.nemotron_proxy import MODEL
from responses_api_agents.promptbio_bench.runtime import (
    DATA_REVISION,
    SCORER_REVISION,
    TOOLS,
    parse_score,
    safe_task_id,
    stage_inputs,
)


def assistant_request_message(message: dict) -> dict:
    """Preserve assistant semantics while excluding response-only null metadata.

    The durable guarded transport retains the original response. Non-null fields
    that the request protocol cannot represent must fail rather than disappear.
    """
    supported = NeMoGymChatCompletionAssistantMessageParam.__annotations__
    projected = {}
    for key, value in message.items():
        if key in supported:
            projected[key] = value
        elif value is None:
            continue
        elif (
            key == "provider_specific_fields"
            and isinstance(value, dict)
            and all(item is None for item in value.values())
        ):
            continue
        else:
            raise ValueError(f"Unsupported non-null assistant response field: {key}")
    return TypeAdapter(NeMoGymChatCompletionAssistantMessageParam).validate_python(projected)


def validate_image(image: str | None) -> None:
    if not image or not re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", image):
        raise ValueError("PromptBio requires a qualified immutable image digest")


def remove_socket_directory(path: Path) -> None:
    original = sys.exception()
    try:
        path.rmdir()
    except OSError as error:
        if original is None:
            raise
        original.add_note(f"PromptBio socket directory cleanup failed: {error}")


def source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    result = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("managed_runtime.py", "guarded_scorer.py", "runtime.py", "nemotron_proxy.py")
    }
    from nemo_gym import guarded_chat_transport

    result["guarded_chat_transport.py"] = hashlib.sha256(
        Path(guarded_chat_transport.__file__).read_bytes()
    ).hexdigest()
    return result


def tree_manifest(directory: Path, *, materialize_symlinks: bool = False) -> dict[str, str]:
    if materialize_symlinks:
        for path in directory.rglob("*"):
            if path.is_symlink() and path.is_file():
                continue
            if not (stat.S_ISREG(path.lstat().st_mode) or stat.S_ISDIR(path.lstat().st_mode)):
                raise ValueError("Unexpected pinned task source file type")
    else:
        validate_saved_outputs(directory)
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def validate_saved_outputs(directory: Path) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("PromptBio outputs must be a real directory")
    for path in directory.rglob("*"):
        if not (stat.S_ISREG(path.lstat().st_mode) or stat.S_ISDIR(path.lstat().st_mode)):
            raise ValueError("PromptBio saved outputs contain symlink or special file")


async def solve_managed(
    task_dir: Path,
    output_root: Path,
    *,
    endpoint: str,
    rollout_id: str,
    model: str,
    image: str,
    max_steps: int = 100,
    max_tokens: int = 16384,
) -> Path:
    if model != MODEL or "/ng-rollout/" not in endpoint:
        raise ValueError("PromptBio requires exact Nemotron through native guarded capture")
    validate_image(image)
    task_id = safe_task_id(task_dir.name)
    signature = {
        "model": model,
        "task": task_id,
        "dataset": DATA_REVISION,
        "image": image,
        "max_steps": max_steps,
        "max_tokens": max_tokens,
        "cpus": 2,
        "memory_mib": 4096,
        "rollout_id": rollout_id,
        "implementation": source_hashes(),
    }
    lineage = uuid.uuid4().hex
    run = output_root / task_id / lineage
    run.mkdir(parents=True, exist_ok=False)  # No automatic shell recreation/resume.
    task = stage_inputs(task_dir, run / "workspace")
    signature["task_json_sha256"] = hashlib.sha256((task_dir / "task.json").read_bytes()).hexdigest()
    signature["staged_files"] = {
        str(p.relative_to(run / "workspace")): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((run / "workspace").rglob("*"))
        if p.is_file()
    }
    _publish(run / "identity.json", {"lineage": lineage, "signature": signature})
    digest = configuration_digest(signature)
    started = time.monotonic()
    deadline = started + int(task.get("timeout_seconds", 3600))
    messages = [
        {
            "role": "system",
            "content": "Solve the task using shell. Work in /workspace. "
            "Read task.json and data/. Write final requested output files into /workspace/outputs/. "
            "No reference solution is available. Finish with a brief summary.",
        },
        {"role": "user", "content": json.dumps(task)},
    ]
    sandbox = AsyncSandbox(
        {
            "docker": {
                "create": {
                    "network": "none",
                    "pids_limit": 256,
                    "cap_drop": ["ALL"],
                    "security_opt": ["no-new-privileges"],
                }
            }
        },
        SandboxSpec(
            image=image,
            ttl_s=int(task.get("timeout_seconds", 3600)) + 60,
            workdir="/workspace",
            resources=SandboxResources(cpu=2, memory_mib=4096),
            metadata={"workload": "promptbio-solver", "run": lineage},
            provider_options={
                "volumes": [f"{(run / 'workspace').resolve()}:/workspace"],
                "run_args": ["--user", f"{os.getuid()}:{os.getgid()}"],
            },
        ),
    )
    status, failure, usage = "failed", None, []
    # UDS stays outside the solver volume; only host-side controller can use it.
    socket_dir = Path("/tmp") / ("gym-promptbio-" + lineage)
    socket_dir.mkdir(mode=0o700)
    token = uuid.uuid4().hex
    try:
        async with _owned_sandbox(sandbox, run / "solver-lifecycle.json"):
            async with bridge(
                socket_dir / "model.sock",
                endpoint=endpoint,
                token=token,
                lineage=lineage,
                digest=digest,
                namespace="promptbio_solver",
                expected_model=model,
            ):
                transport = GuardedChatTransport(
                    socket_path=str(socket_dir / "model.sock"),
                    token=token,
                    lineage=lineage,
                    digest=digest,
                    namespace="promptbio_solver",
                    model=model,
                    output=run / "calls",
                )
                async with httpx.AsyncClient(transport=transport) as client:
                    for _ in range(max_steps):
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("Task timeout")
                        payload = {
                            "model": model,
                            "messages": messages,
                            "tools": TOOLS,
                            "temperature": 0,
                            "max_tokens": max_tokens,
                        }
                        async with asyncio.timeout(min(600, remaining)):
                            response = await client.post("http://gym-internal/v1/chat/completions", json=payload)
                            result = response.json()
                        usage.append(result.get("usage"))
                        message = assistant_request_message(result["choices"][0]["message"])
                        messages.append(message)
                        _publish(run / "trajectory.json", {"messages": messages})
                        if result["choices"][0].get("finish_reason") == "length":
                            raise TimeoutError("Solver output token budget exhausted")
                        calls = message.get("tool_calls") or []
                        if not calls:
                            status = "completed"
                            break
                        for call in calls:
                            try:
                                if call["function"]["name"] != "shell":
                                    raise ValueError("Unknown tool")
                                script = json.loads(call["function"]["arguments"])["command"]
                                remaining = deadline - time.monotonic()
                                if remaining <= 0:
                                    raise TimeoutError("Task timeout")
                                seconds = max(1, int(min(300, remaining)))
                                execution = await sandbox.exec(
                                    f"timeout {seconds} bash -lc {shlex.quote(script)}",
                                    timeout_s=min(310, remaining + 1),
                                )
                                if execution.error_type and execution.error_type != "timeout":
                                    raise RuntimeError("Sandbox infrastructure failure: " + str(execution.error_type))
                                if execution.return_code:
                                    value = f"Command failed: {(execution.stderr or '')[-2000:]}"
                                else:
                                    value = execution.stdout or ""
                            except (ValueError, KeyError) as exc:
                                value = str(exc)
                            messages.append({"role": "tool", "tool_call_id": call["id"], "content": value[-30000:]})
                    else:
                        raise TimeoutError("Agent step limit")
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        status = (
            "model_budget_exhausted"
            if isinstance(exc, TimeoutError)
            and str(exc) in ("Task timeout", "Agent step limit", "Solver output token budget exhausted")
            else "failed"
        )
        if status == "model_budget_exhausted":
            lifecycle = run / "solver-lifecycle.json"
            if not lifecycle.exists() or not json.loads(lifecycle.read_text()).get("cleanup_confirmed"):
                status = "failed"
                raise RuntimeError("Solver budget ended but sandbox cleanup was not confirmed") from exc
        else:
            raise
    finally:
        _publish(
            run / "receipt.json",
            {
                "signature": signature,
                "status": status,
                "failure": failure,
                "elapsed_seconds": time.monotonic() - started,
                "usage": usage,
            },
        )
        _publish(run / "trajectory.json", {"messages": messages})
        remove_socket_directory(socket_dir)
    return run


async def validate_creator_source(source: Path) -> None:
    """Reject non-committed checkout files before exposing a trusted creator tree."""
    provider = LocalProvider()
    checks = [
        ("rev-parse", "HEAD"),
        ("status", "--porcelain", "--untracked-files=all"),
        ("ls-files", "--others", "--ignored", "--exclude-standard"),
    ]
    async with scoring_workspace(provider) as handle:
        for index, arguments in enumerate(checks):
            result = await provider.exec(
                handle,
                shlex.join(["git", "-C", str(source), *arguments]),
                timeout_s=30,
                terminate_background_children=True,
            )
            if result.return_code or result.error_type:
                raise ValueError("PromptBio creator source preflight failed")
            text = (result.stdout or "").strip()
            if (index == 0 and text != SCORER_REVISION) or (index != 0 and text):
                raise ValueError("PromptBio requires a clean pinned creator checkout, including no ignored files")


async def snapshot_creator(source: Path, output: Path) -> tuple[Path, dict[str, str]]:
    """Expose only pinned committed files, never Git configuration/hooks or ignored state."""
    destination = output / "creator-source"
    archive = output / "creator.tar"
    provider = LocalProvider()
    async with scoring_workspace(provider) as handle:
        result = await provider.exec(
            handle,
            shlex.join(
                [
                    "git",
                    "-C",
                    str(source),
                    "archive",
                    "--format=tar",
                    "--output=" + str(archive.resolve()),
                    SCORER_REVISION,
                ]
            ),
            timeout_s=30,
            terminate_background_children=True,
        )
        if result.return_code or result.error_type:
            raise ValueError("Pinned creator source archive failed")
    destination.mkdir()
    with tarfile.open(archive) as contents:
        for member in contents.getmembers():
            if not (member.isfile() or member.isdir()):
                raise ValueError("Creator archive contains unsupported non-regular file")
        contents.extractall(destination, filter="data")
    return destination, tree_manifest(destination)


async def grade_managed(
    task_dir: Path,
    runs: list[Path],
    *,
    scorer_dir: Path,
    python: str,
    output: Path,
    endpoint: str,
    rollout_id: str,
    scorer_image: str | None = None,
    scorer_memory_mib: int = 4096,
) -> list[float]:
    """One owned creator invocation preserves shared Detect/Recommend across runs."""
    validate_image(scorer_image)
    await validate_creator_source(scorer_dir)
    if scorer_memory_mib <= 0:
        raise ValueError("Invalid scorer memory bound")
    for run in runs:
        validate_saved_outputs(run / "workspace/outputs")
    output.mkdir(parents=True, exist_ok=False)
    creator_copy, creator_manifest = await snapshot_creator(scorer_dir, output)
    lineage = uuid.uuid4().hex
    signature = {
        "creator_files": creator_manifest,
        "scorer": SCORER_REVISION,
        "scoring_variant": "candidate-membership-v1",
        "dataset": DATA_REVISION,
        "model": MODEL,
        "rollout_id": rollout_id,
        "run_receipts": [hashlib.sha256((r / "receipt.json").read_bytes()).hexdigest() for r in runs],
        "implementation": source_hashes(),
        "scorer_image": scorer_image,
        "scorer_memory_mib": scorer_memory_mib,
    }
    socket_dir = Path("/tmp") / ("gym-promptbio-judge-" + lineage)
    controller = output / "controller"
    package = controller / "responses_api_agents/promptbio_bench"
    package.mkdir(parents=True)
    shared = controller / "nemo_gym/sandbox"
    shared.mkdir(parents=True)
    for parent in (package, package.parent, shared, shared.parent):
        (parent / "__init__.py").write_text("")
    from nemo_gym import guarded_chat_transport

    shutil.copyfile(Path(guarded_chat_transport.__file__), shared / "guarded_chat_transport.py")
    for name in ("guarded_scorer.py", "runtime.py", "nemotron_proxy.py"):
        shutil.copyfile(Path(__file__).with_name(name), package / name)
    # HF source symlinks are materialized before mounting: no external cache or host path enters the judge.
    task_copy = output / "task-source"
    task_manifest = tree_manifest(task_dir, materialize_symlinks=True)
    shutil.copytree(task_dir, task_copy, symlinks=False)
    if tree_manifest(task_copy) != task_manifest:
        raise ValueError("PromptBio pinned task files changed during snapshot")
    signature["task_files"] = task_manifest
    candidate_root = output / "candidate-inputs"
    candidate_root.mkdir()
    signature["candidate_files"] = []
    candidates = []
    for index, run in enumerate(runs):
        original = run / "workspace/outputs"
        manifest = tree_manifest(original)
        snapshot = candidate_root / str(index)
        shutil.copytree(original, snapshot)
        if tree_manifest(snapshot) != manifest:
            raise ValueError("PromptBio candidate files changed during snapshot")
        candidates.append(snapshot)
        signature["candidate_files"].append(manifest)
    signature["controller_files"] = tree_manifest(controller)
    digest = configuration_digest(signature)
    _publish(output / "identity.json", {"lineage": lineage, "signature": signature})
    request = {
        "scorer_dir": "/creator",
        "creator_revision": SCORER_REVISION,
        "creator_manifest": creator_manifest,
        "task_dir": "/task",
        "runs": [f"/runs/{i}/{r.name}" for i, r in enumerate(runs)],
        "output": "/results",
        "socket_path": "/model/model.sock",
        "token": uuid.uuid4().hex,
        "lineage": lineage,
        "digest": digest,
        "namespace": "promptbio_judge",
        "model": MODEL,
    }
    _publish(output / "request.json", request)
    volumes = [
        f"{creator_copy.resolve()}:/creator:ro",
        f"{creator_copy.resolve()}:/results/creator-source:ro",
        f"{controller.resolve()}:/controller:ro",
        f"{task_copy.resolve()}:/task:ro",
        f"{output.resolve()}:/results",
        f"{socket_dir}:/model:ro",
        f"{controller.resolve()}:/results/controller:ro",
        f"{task_copy.resolve()}:/results/task-source:ro",
        f"{candidate_root.resolve()}:/results/candidate-inputs:ro",
        f"{(output / 'request.json').resolve()}:/results/request.json:ro",
    ]
    volumes += [f"{candidates[i].resolve()}:/runs/{i}/{r.name}/workspace/outputs:ro" for i, r in enumerate(runs)]
    sandbox = AsyncSandbox(
        {
            "docker": {
                "create": {
                    "network": "none",
                    "cap_drop": ["ALL"],
                    "security_opt": ["no-new-privileges"],
                    "pids_limit": 256,
                }
            }
        },
        SandboxSpec(
            image=scorer_image,
            ttl_s=3660,
            workdir="/creator",
            resources=SandboxResources(cpu=2, memory_mib=scorer_memory_mib),
            metadata={"workload": "promptbio-judge", "run": lineage},
            env={
                "PYTHONPATH": "/controller",
                "OPENAI_API_KEY": "task-scoped-proxy",
                "OPENAI_BASE_URL": "http://gym-internal/v1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            provider_options={"volumes": volumes, "run_args": ["--user", f"{os.getuid()}:{os.getgid()}"]},
        ),
    )
    socket_dir.mkdir(mode=0o700)
    try:
        async with bridge(
            socket_dir / "model.sock",
            endpoint=endpoint,
            token=request["token"],
            lineage=lineage,
            digest=digest,
            namespace="promptbio_judge",
            expected_model=MODEL,
        ):
            async with _owned_sandbox(sandbox, output / "judge-lifecycle.json"):
                argv = [
                    python,
                    "/controller/responses_api_agents/promptbio_bench/guarded_scorer.py",
                    "/results/request.json",
                ]
                execution = await sandbox.exec(shlex.join(argv), timeout_s=3600)
                _publish(
                    output / "execution.json",
                    {
                        "return_code": execution.return_code,
                        "stdout": execution.stdout,
                        "stderr": execution.stderr,
                        "error_type": execution.error_type,
                    },
                )
                if execution.return_code != 0 or execution.error_type:
                    raise RuntimeError("PromptBio creator evaluator failed; raw metrics retained, no reward")
    finally:
        remove_socket_directory(socket_dir)
    envelope = json.loads((output / "worker-complete.json").read_text())
    if (
        envelope["implementation_sha256"]
        != hashlib.sha256(Path(__file__).with_name("guarded_scorer.py").read_bytes()).hexdigest()
    ):
        raise ValueError("PromptBio scorer imported source mismatch")
    if (
        envelope.get("guarded_transport_sha256")
        != hashlib.sha256(Path(guarded_chat_transport.__file__).read_bytes()).hexdigest()
    ):
        raise ValueError("PromptBio shared guard imported source mismatch")
    if envelope.get("paired_runs") != len(runs) or envelope.get("scorer_revision") != SCORER_REVISION:
        raise ValueError("PromptBio creator pipeline completion identity mismatch")
    scores = [parse_score(output / f"{task_dir.name}_{run.name}.json") for run in runs]
    _publish(output / "grading-receipt.json", signature)
    return scores


async def download_managed(task_id: str, cache: Path, output: Path) -> Path:
    """Settle pinned task acquisition in an owned subprocess, not an unkillable thread."""

    safe_task_id(task_id)
    provider = LocalProvider()
    command = "from pathlib import Path; from responses_api_agents.promptbio_bench.runtime import download_task; import sys; print(download_task(sys.argv[1], Path(sys.argv[2])))"
    argv = [
        "env",
        "-i",
        "PATH=" + os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH=" + str(Path(__file__).resolve().parents[2]),
        sys.executable,
        "-c",
        command,
        task_id,
        str(cache),
    ]
    async with scoring_workspace(provider) as handle:
        execution = await provider.exec(handle, shlex.join(argv), timeout_s=1800, terminate_background_children=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        _publish(
            output,
            {
                "return_code": execution.return_code,
                "stdout": execution.stdout,
                "stderr": execution.stderr,
                "error_type": execution.error_type,
            },
        )
        if execution.return_code or execution.error_type:
            raise RuntimeError("Pinned PromptBio task acquisition failed")
    path = Path((execution.stdout or "").strip().splitlines()[-1])
    if path.name != task_id or not (path / "task.json").is_file():
        raise ValueError("Task acquisition returned unexpected source")
    return path
