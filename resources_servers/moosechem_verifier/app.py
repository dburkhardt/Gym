# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference-owning MOOSE resource; public solver runs in a different sandbox."""

import asyncio
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr

from benchmarks.moosechem.isolated_execution import verifier
from benchmarks.moosechem.native import MODEL
from nemo_gym.base_resources_server import BaseResourcesServerConfig, SimpleResourcesServer
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict, get_global_config_dict
from nemo_gym.rollout_correlation import current_rollout_id
from nemo_gym.server_utils import rollout_path_prefix


class Config(BaseResourcesServerConfig):
    output_root: Path
    upstream_repo: Path
    isolated_runtime_image: str
    output_token_budget: Literal[8192, 16384] = 8192
    native_timeout_seconds: float = Field(default=14400, gt=0)
    model_server: ModelServerRef = Field(
        default_factory=lambda: ModelServerRef(type="responses_api_models", name="policy_model")
    )


class NativeRequest(BaseModel):
    artifact_dir: Path
    identity_sha256: str


class Server(SimpleResourcesServer):
    config: Config
    _semaphore: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(1))

    def setup_webserver(self):
        app = super().setup_webserver()
        app.post("/verify_native")(self.verify_native)
        return app

    async def verify(self, body):
        raise ValueError("Use the source-bound native verifier route; no implicit judge replay")

    async def verify_native(self, body: NativeRequest):
        c = get_global_config_dict()
        lineage = current_rollout_id()
        if not lineage or not c.get("provider_dispatch_guard") or c.get("policy_model_name") != MODEL:
            raise ValueError("Exact guarded verifier rollout required")
        model = get_first_server_config_dict(c, self.config.model_server.name)
        endpoint = f"http://{model['host']}:{model['port']}" + rollout_path_prefix(lineage) + "/v1"
        async with self._semaphore:
            return await verifier(self.config, body.artifact_dir, body.identity_sha256, endpoint, lineage)


if __name__ == "__main__":
    Server.run_webserver()
