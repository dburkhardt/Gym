# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discoverable planning-only WMDP adapter; canonical likelihood transport is missing."""

from fastapi import Request
from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.conditional_likelihood import LikelihoodUnavailable
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming


class RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict


class Agent(SimpleResponsesAPIAgent):
    config: BaseResponsesAPIAgentConfig

    async def responses(self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        """Reject generated-answer substitution before any model request."""
        raise LikelihoodUnavailable("WMDP requires qualified raw conditional continuation likelihoods, not generation")

    async def run(self, request: Request, body: RunRequest) -> BaseVerifyResponse:
        """Fail closed until an exact tokenizer-bound likelihood backend is integrated."""
        raise LikelihoodUnavailable(
            "No admitted WMDP conditional-likelihood backend; this adapter supports planning only"
        )


if __name__ == "__main__":
    Agent.run_webserver()
