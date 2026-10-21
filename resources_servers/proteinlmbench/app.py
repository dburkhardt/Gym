# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native single-turn ProteinLMBench verifier."""

from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.completion_outcome import with_completion_outcome
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.proteinlmbench.grading import grade
from resources_servers.proteinlmbench.task_data import TaskData


class ProteinLMBenchConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class ProteinLMBenchRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: TaskData


class ProteinLMBenchResponse(BaseVerifyResponse, ProteinLMBenchRequest):
    predicted_answer: int | None = None
    strict_answer: int | None = None
    strict_correct: bool = False
    out_of_range: bool = False


class ProteinLMBenchVerifier:
    async def verify(self, body: ProteinLMBenchRequest) -> ProteinLMBenchResponse:
        result = grade(body.response.output_text or "", **body.verifier_metadata.model_dump())
        return ProteinLMBenchResponse(
            **with_completion_outcome({**body.model_dump(), **result}, response=body.response)
        )


class ProteinLMBenchResourcesServer(ProteinLMBenchVerifier, SimpleResourcesServer):
    config: ProteinLMBenchConfig
    ray_enabled = False


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=ProteinLMBenchVerifier,
    request_model=ProteinLMBenchRequest,
    cases_path=Path(__file__).parent / "tests" / "fixture_cases.jsonl",
)


if __name__ == "__main__":
    ProteinLMBenchResourcesServer.run_webserver()
