# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-row strict extraction; sequential creator fallback is a separate regrade."""

import re
from typing import ClassVar

from pydantic import ConfigDict

from benchmarks.scienceqa.creator_recipe import PROTOCOL
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)


class Config(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class VerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    task_id: str
    verifier_metadata: dict


class VerifyResponse(BaseVerifyResponse, VerifyRequest):
    scoring_details: dict


class Server(SimpleResourcesServer):
    config: Config
    ray_enabled = False

    async def verify(self, body: VerifyRequest) -> VerifyResponse:
        meta = body.verifier_metadata
        if meta["protocol"] != PROTOCOL:
            raise ValueError("Unexpected ScienceQA protocol")
        choices = meta["choices"]
        answer = meta["answer"]
        if type(answer) is not int or not 0 <= answer < len(choices) <= 5:
            raise ValueError("Invalid ScienceQA reference")
        matches = re.findall(r"The answer is ([A-Z]).", body.response.output_text.strip())
        valid = len(matches) == 1 and matches[0] in "ABCDE"[: len(choices)]
        strict = float(valid and "ABCDE".index(matches[0]) == answer)
        details = {
            "protocol": PROTOCOL,
            "strict_accuracy": strict,
            "creator_raw_accuracy": strict if valid else None,
            "fallback_randomized": not valid,
            "sequential_fallback_status": "not_needed" if valid else "uncomputed_pending_ordered_saved_output_regrade",
            "measurement_valid": None if valid else False,
            "qualification": "Reference correctness not independently reviewed"
            if valid
            else "Ordered subset RNG requires prior outputs; strict zero is not a computed creator fallback score",
        }
        payload = body.model_dump() | {"reward": strict, "mask_sample": not valid, "scoring_details": details}
        return VerifyResponse.model_validate(payload)


if __name__ == "__main__":
    Server.run_webserver()
