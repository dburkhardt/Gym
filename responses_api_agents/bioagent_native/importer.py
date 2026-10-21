# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect a completed creator Verifiers rollout into Gym without repeated inference."""

import json
import time
from pathlib import Path
from uuid import uuid4

from fastapi import Request, Response

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming


MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


class Config(BaseResponsesAPIAgentConfig):
    trace_path: Path
    provider_trace: Path


class RunRequest(BaseRunRequest):
    verifier_metadata: dict


class VerifyResponse(BaseVerifyResponse):
    native_metrics: dict
    imported_saved_execution: bool = True
    synthetic_summary_response: bool = True


class Agent(SimpleResponsesAPIAgent):
    config: Config

    async def responses(
        self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        raise NotImplementedError("Use /run to collect completed native rollout")

    async def run(self, request: Request, body: RunRequest) -> VerifyResponse:
        rows = [json.loads(x) for x in self.config.trace_path.read_text().splitlines()]
        selected = [r for r in rows if r["task"]["data"]["task_id"] == body.verifier_metadata["id"]]
        assert len(selected) == 1
        native = selected[0]
        assert native["is_completed"] and not native["errors"]
        inputs = body.responses_create_params.model_dump()["input"]
        assert (
            len(inputs) == 1
            and inputs[0]["role"] == "user"
            and inputs[0]["content"] == native["task"]["data"]["prompt"]
        )
        calls = [json.loads(x) for x in self.config.provider_trace.read_text().splitlines()]
        calls = [x for x in calls if native["timing"]["start"] <= x["time"] <= native["timing"]["scoring"]["end"]]
        assert calls and all(x["response"]["model"] == MODEL for x in calls)
        reward = native["rewards"]["results_match"]
        assert reward in (0, 1)
        text = json.dumps({"native_metrics": native["metrics"], "native_assessment": native["info"]["assessment"]})
        response = NeMoGymResponse.model_validate(
            {
                "id": "resp_" + uuid4().hex,
                "object": "response",
                "created_at": time.time(),
                "model": MODEL,
                "status": "completed",
                "output": [
                    {
                        "id": "msg_" + uuid4().hex,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "annotations": [], "text": text}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        return VerifyResponse(
            **body.model_dump(),
            response=response,
            reward=reward,
            native_metrics={
                "metrics": native["metrics"],
                "rewards": native["rewards"],
                "actual_model_calls": len(calls),
                "native_trace": str(self.config.trace_path),
                "provider_trace": str(self.config.provider_trace),
                "protocol": "creator_Verifiers_OpenCode_Nemotron_all_roles_buffered_SSE",
            },
        )


if __name__ == "__main__":
    Agent.run_webserver()
