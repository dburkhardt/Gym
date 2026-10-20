# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SciCode resources server.

Runs the agent's accumulated per-sub-step Python solutions against each sub-step's test cases
(targets loaded from test_data.h5) and returns a binary reward: 1.0 iff every sub-step passes.
Per-sub-step counts are also returned so sub-step accuracy can be computed downstream.

Each sub-step executes in an owned, network-disabled scientific sandbox. Generated code
never inherits this server's interpreter, environment variables, or provider credentials.
"""

import asyncio
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import ConfigDict, Field
from scicode_integration.runner import build_test_program, sanitize_test

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.scicode.managed_execution import execute_substep


# Agent sentinel for a sub-step it could not generate (ran out of context); always fails.
_OUT_OF_CONTEXT = "_ran_out_of_context_"


class ScicodeResourcesServerConfig(BaseResourcesServerConfig):
    num_processes: int = Field(default=2, ge=1)
    sandbox_image: str | None = None
    sandbox_cpu: float = Field(default=1.0, gt=0)
    sandbox_memory_mib: int = Field(default=4096, ge=128)
    execution_artifact_dir: str = "artifacts/scicode-execution"
    test_data_sha256: str = "48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890"
    # Per-sub-step execution timeout
    timeout_secs: float = Field(default=30.0, gt=0)
    # Local path to SciCode's test_data.h5 (staged manually)
    test_data_fpath: Optional[str] = None


class ScicodeRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    problem_id: str
    sub_steps: List[dict]
    # {"<problem_id>.<step>": accumulated_code} produced by the agent.
    solutions: Optional[Dict[str, str]] = None


class ScicodeVerifyRequest(ScicodeRunRequest, BaseVerifyRequest):
    pass


class ScicodeVerifyResponse(BaseVerifyResponse):
    # Retain the agent's compact accounting records through verification. None
    # distinguishes historical final-step-only rollouts from whole-problem usage.
    token_usage_version: Optional[int] = None
    step_usage: Optional[List[dict]] = None
    # Declared so it survives into the rollout output (identifies the problem); the request's
    # sub_steps/solutions are intentionally not carried through to keep rollout rows small.
    problem_id: str = ""
    step_results: List[bool] = []
    step_execution: List[dict] = []
    num_steps_passed: int = 0
    num_steps_total: int = 0
    problem_accuracy: bool = False
    # Per-rollout sub-step pass fraction. As a numeric verify-response field it
    # gets the full generic statistics (mean/std/min/max/median) from the
    # RewardProfiler, which the aggregate-only pooled scalar cannot provide.
    subtask_accuracy: float = 0.0


class ScicodeResourcesServer(SimpleResourcesServer):
    config: ScicodeResourcesServerConfig

    def model_post_init(self, context):
        self._semaphore = asyncio.Semaphore(value=self.config.num_processes)
        self._validated_data = None

    def _resolve_test_data(self) -> str:
        if not self.config.test_data_fpath:
            raise RuntimeError(
                "test_data_fpath is not configured. Stage SciCode's test_data.h5 and set "
                "test_data_fpath (see benchmarks/scicode/README.md)."
            )
        path = Path(self.config.test_data_fpath).expanduser()
        # Resolve relative paths against the Gym root, since the server's cwd is its own dir.
        if not path.is_absolute():
            path = PARENT_DIR / path
        if not path.is_file():
            raise RuntimeError(
                f"SciCode test_data.h5 not found at {path}. Download and stage it "
                "(see benchmarks/scicode/README.md) before running."
            )
        stat = path.stat()
        identity = (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if self._validated_data != identity:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != self.config.test_data_sha256:
                raise ValueError("SciCode test_data.h5 SHA256 mismatch; refusing scoring")
            self._validated_data = identity
        return str(path)

    async def verify(self, body: ScicodeVerifyRequest) -> ScicodeVerifyResponse:
        solutions = body.solutions or {}
        # Score only sub-steps the agent produced a solution for. Sub-steps absent from solutions
        # (prefilled steps) are excluded from the denominator entirely; out-of-context sentinels
        # are present and counted as failures.
        scored = [i for i in range(len(body.sub_steps)) if f"{body.problem_id}.{i + 1}" in solutions]
        if not scored:
            return ScicodeVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                step_results=[],
                num_steps_passed=0,
                num_steps_total=0,
                problem_accuracy=False,
            )

        h5_path = await asyncio.to_thread(self._resolve_test_data)
        if not self.config.sandbox_image:
            raise RuntimeError("SciCode sandbox_image must name the pinned Python3.12 scientific runtime")
        artifacts = Path(self.config.execution_artifact_dir).expanduser()
        if not artifacts.is_absolute():
            artifacts = PARENT_DIR / artifacts

        execution_details = {}

        async def _run_substep(i: int) -> bool:
            sub_step = body.sub_steps[i]
            code = solutions[f"{body.problem_id}.{i + 1}"]
            if not code or code == _OUT_OF_CONTEXT:
                return False
            sanitized = [sanitize_test(tc) for tc in sub_step["test_cases"]]
            program = build_test_program(code, "/data/test_data.h5", sub_step["step_number"], sanitized)
            async with self._semaphore:
                result = await execute_substep(
                    program=program,
                    data_path=Path(h5_path),
                    image=self.config.sandbox_image,
                    artifact_dir=artifacts,
                    timeout_secs=self.config.timeout_secs,
                    cpu=self.config.sandbox_cpu,
                    memory_mib=self.config.sandbox_memory_mib,
                )
            execution_details[i] = {"step_number": sub_step["step_number"], **result}
            return bool(result["passed"])

        step_results = list(await asyncio.gather(*[_run_substep(i) for i in scored]))
        num_passed = sum(step_results)
        all_passed = num_passed == len(scored)

        return ScicodeVerifyResponse(
            **body.model_dump(),
            reward=1.0 if all_passed else 0.0,
            step_results=step_results,
            step_execution=[execution_details[i] for i in scored if i in execution_details],
            num_steps_passed=num_passed,
            num_steps_total=len(scored),
            problem_accuracy=all_passed,
            subtask_accuracy=num_passed / len(scored),
        )


if __name__ == "__main__":
    ScicodeResourcesServer.run_webserver()
