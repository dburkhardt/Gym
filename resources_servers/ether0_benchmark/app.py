# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Gym verifier; references never enter model messages."""

import json
import os
import shlex
import sys
from pathlib import Path
from typing import ClassVar

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.sandbox.providers.local.provider import LocalProvider
from nemo_gym.sandbox.scoring_workspace import scoring_workspace


class Config(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class Request(BaseVerifyRequest):
    verifier_metadata: dict


class Response(BaseVerifyResponse, Request):
    diagnostics: dict


class Verifier:
    async def verify(self, body: Request) -> Response:
        try:
            provider = LocalProvider(concurrency=2)
            async with scoring_workspace(provider) as handle:
                request = handle.raw["workspace"] / "request.json"
                request.write_text(json.dumps({"text": body.response.output_text or "", **body.verifier_metadata}))
                source_root = Path(__file__).resolve().parents[2]
                worker = Path(__file__).with_name("score_worker.py")
                command = shlex.join([sys.executable, str(worker.resolve())]) + " < " + shlex.quote(str(request))
                runtime = await provider.exec(
                    handle,
                    command,
                    timeout_s=120,
                    env={
                        "PYTHONPATH": os.pathsep.join([str(source_root), *sys.path]),
                        "NEMO_GYM_EXTRA_ROOTS": str(source_root),
                    },
                )
                if runtime.error_type == "timeout":
                    raise RuntimeError("Creator scorer subprocess exceeded 120 seconds")
                if runtime.return_code:
                    raise RuntimeError(f"Creator scorer subprocess exited {runtime.return_code}: {runtime.stderr}")
                records = [
                    line.removeprefix("ETHER0_RESULT_JSON:")
                    for line in (runtime.stdout or "").splitlines()
                    if line.startswith("ETHER0_RESULT_JSON:")
                ]
                if len(records) != 1:
                    raise ValueError("Creator worker must return exactly one framed JSON record")
                result = json.loads(records[0])
        except (RuntimeError, OSError, ValueError) as exc:
            return Response(
                **body.model_dump(),
                reward=0.0,
                mask_sample=True,
                failure_kind="ether0_benchmark:grader_unavailable",
                failure_reason=str(exc),
                diagnostics={
                    "reward_is_serialization_placeholder": True,
                    "denominator_policy": "Retain attempt; exclude placeholder from scientific performance.",
                    "oracle_error": str(exc),
                },
            )
        return Response(**body.model_dump(), **result)


class Server(Verifier, SimpleResourcesServer):
    config: Config
    ray_enabled = False


if __name__ == "__main__":
    Server.run_webserver()
