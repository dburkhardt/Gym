# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym execution adapter for independently staged BixBench3 cases."""

import asyncio
import hashlib
import json
import time
from pathlib import Path

from fastapi import HTTPException, Request, Response
from pydantic import PrivateAttr

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.bixbench3.runtime import read_artifact_score


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
        raise HTTPException(status_code=400, detail="BixBench3 owns its Inspect episode; use /run")

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
                expected = hashlib.sha256(settings_path.read_bytes()).hexdigest()
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
                    )
                    try:
                        code = await asyncio.wait_for(process.wait(), timeout=self.config.timeout_seconds)
                    except (TimeoutError, asyncio.CancelledError):
                        process.terminate()
                        try:
                            await asyncio.wait_for(process.wait(), timeout=15)
                        except TimeoutError:
                            process.kill()
                            await process.wait()
                        raise
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
