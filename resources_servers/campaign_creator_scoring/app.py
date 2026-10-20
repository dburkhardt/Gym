# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Gym verifier delegating to pinned creator scoring in isolated processes."""
import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import ClassVar
from nemo_gym.base_resources_server import BaseResourcesServerConfig, BaseVerifyRequest, BaseVerifyResponse, ReverifyMode, SimpleResourcesServer
from benchmarks.scientific_agent_campaign.deterministic import score

_POOL = ProcessPoolExecutor(max_workers=2)

class Config(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    benchmark: str
    upstream_repo: str

class VerifyRequest(BaseVerifyRequest):
    uuid: str
    verifier_metadata: dict

class VerifyResponse(BaseVerifyResponse, VerifyRequest):
    scoring_details: dict

class CreatorResourcesServer(SimpleResourcesServer):
    config: Config
    ray_enabled = False

    async def verify(self, body: VerifyRequest) -> VerifyResponse:
        details = await asyncio.get_running_loop().run_in_executor(
            _POOL, score, self.config.benchmark, Path(self.config.upstream_repo),
            {'uuid': body.uuid, 'verifier_metadata':body.verifier_metadata}, body.response.output_text or '')
        if details.get('reward') is None:
            raise ValueError('This task has no upstream automatic score; do not convert to zero.')
        return VerifyResponse(**body.model_dump(by_alias=True), reward=details['reward'], scoring_details=details)

if __name__ == '__main__':
    CreatorResourcesServer.run_webserver()
