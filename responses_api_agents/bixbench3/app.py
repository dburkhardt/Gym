# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym execution adapter for independently staged BixBench3 cases."""

import asyncio
import hashlib
import json
import os
import signal
import time
from pathlib import Path

from fastapi import HTTPException, Request, Response
from pydantic import PrivateAttr

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.bixbench3.runtime import read_artifact_score, run_fingerprint


async def terminate_process_tree(process: asyncio.subprocess.Process) -> None:
    """Stop a session leader and descendants, including children ignoring TERM."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=15)
    except TimeoutError:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def cleanup_run_containers(run_root: Path) -> None:
    """Remove only containers and networks labeled for this exact run directory."""
    label = "gym.bixbench3.run=" + hashlib.sha256(str(run_root).encode()).hexdigest()[:24]
    for kind, query, delete in [
        ("container", ["docker", "ps", "-aq"], ["docker", "rm", "-f"]),
        ("network", ["docker", "network", "ls", "-q"], ["docker", "network", "rm"]),
    ]:
        process = await asyncio.create_subprocess_exec(
            *query, "--filter", f"label={label}", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        if process.returncode:
            raise RuntimeError(f"Cannot enumerate run {kind}s: {stderr.decode(errors='replace')}")
        ids = stdout.decode().split()
        if ids:
            process = await asyncio.create_subprocess_exec(
                *delete, *ids, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
            if process.returncode:
                raise RuntimeError(f"Cannot clean run {kind}s: {stderr.decode(errors='replace')}")


class BixBench3Config(BaseResponsesAPIAgentConfig):
    """Operator-owned configuration; secrets live only in referenced host files."""

    settings_root: str
    runtime_python: str
    timeout_seconds: int = 87600


class BixBench3Request(BaseRunRequest):
    task_id: str
    settings_file: str


class BixBench3Result(BaseVerifyResponse):
    task_id: str
    grade_path: str
    inspect_log_dir: str
    process_judging: str = "not_run"


class BixBench3Agent(SimpleResponsesAPIAgent):
    config: BixBench3Config
    _semaphore: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(1))

    async def responses(
        self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        raise NotImplementedError("BixBench3 owns its Inspect episode; use /run")

    async def run(self, body: BixBench3Request = Body(), request: Request = None) -> BixBench3Result:
        """Execute one staged case; only the host-side artifact grade is reward."""
        async with self._semaphore:
            root = Path(self.config.settings_root).resolve(strict=True)
            settings_path = (root / body.settings_file).resolve(strict=True)
            if not settings_path.is_relative_to(root):
                raise HTTPException(status_code=400, detail="Settings path escapes configured root")
            settings = json.loads(settings_path.read_text())
            if settings["task_id"] != body.task_id:
                raise HTTPException(status_code=400, detail="Settings task ID mismatch")
            run_root = Path(settings["run_root"]).resolve(strict=True)
            grade_path = run_root / "grade.json"
            if grade_path.exists():
                expected = await asyncio.to_thread(run_fingerprint, settings_path)
                receipt = run_root / "settings-sha256.txt"
                if not receipt.is_file() or receipt.read_text().strip() != expected:
                    raise HTTPException(status_code=409, detail="Existing result does not match these run settings")
            if not grade_path.exists():
                with (run_root / "gym-adapter.log").open("ab") as log:
                    process = await asyncio.create_subprocess_exec(
                        self.config.runtime_python,
                        str(Path(__file__).with_name("runtime.py")),
                        "--settings",
                        str(settings_path),
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                    )
                    try:
                        code = await asyncio.wait_for(process.wait(), timeout=self.config.timeout_seconds)
                    except (TimeoutError, asyncio.CancelledError):
                        await terminate_process_tree(process)
                        raise
                    finally:
                        await cleanup_run_containers(run_root)
                if code:
                    raise HTTPException(
                        status_code=502, detail=f"BixBench3 execution failed; see {run_root / 'gym-adapter.log'}"
                    )
            score = read_artifact_score(grade_path)
            response = NeMoGymResponse(
                id=f"bixbench3-{body.task_id}",
                created_at=time.time(),
                model=settings["model"],
                object="response",
                output=[],
                parallel_tool_calls=False,
                tools=[],
                tool_choice="auto",
            )
            return BixBench3Result(
                responses_create_params=body.responses_create_params,
                response=response,
                reward=score,
                task_id=body.task_id,
                grade_path=str(grade_path),
                inspect_log_dir=str(run_root / "inspect_logs"),
            )


if __name__ == "__main__":
    BixBench3Agent.run_webserver()
