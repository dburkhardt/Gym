# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit AIRS failure-policy variant: never repeat a failed provider call."""

import json
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.base import LLMResponse

from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import NemoGymLLM
from responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym import Terminus2NemoGym


class NoRetryNemoGymLLM(NemoGymLLM):
    failure_receipt_path: Path | None = None

    async def call(
        self, prompt: str, message_history: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> LLMResponse:
        try:
            return await NemoGymLLM.call.__wrapped__(self, prompt, message_history, **kwargs)
        except Exception as exc:
            if self.failure_receipt_path is not None:
                self.failure_receipt_path.parent.mkdir(parents=True, exist_ok=True)
                self.failure_receipt_path.write_text(
                    json.dumps({"exception_type": type(exc).__name__, "message": str(exc), "retry_performed": False})
                )
            raise


class AIRSNoRetryAgent(Terminus2NemoGym):
    """Keep the scientific scaffold, but disable both nested retry decorators."""

    def __init__(
        self, logs_dir: Path, model_name: str | None = None, api_base: str | None = None, **kwargs: Any
    ) -> None:
        if model_name != "nvidia/nvidia/nemotron-3.5-super-vl" or api_base is None or kwargs.get("llm") is not None:
            raise ValueError("AIRS requires its exact Nemotron model and scoped model server")
        llm = NoRetryNemoGymLLM(
            model_name=model_name,
            api_base=api_base,
            collect_rollout_details=kwargs.get("collect_rollout_details", False),
            model_info=kwargs.get("model_info"),
            responses_create_params=kwargs.get("responses_create_params"),
            timeout_sec=kwargs.get("nemo_model_server_timeout_sec", 120.0),
            dispatch_state_path=logs_dir / "gym-call-state.json" if "/ng-rollout/" in api_base else None,
            dispatch_config={"model": model_name, "harness": "AIRSNoRetryAgent", **kwargs},
        )
        llm.failure_receipt_path = logs_dir / "llm-call-failure.json"
        super().__init__(logs_dir=logs_dir, model_name=model_name, api_base=api_base, llm=llm, **kwargs)

    async def _query_llm(self, *args: Any, **kwargs: Any) -> LLMResponse:
        return await Terminus2._query_llm.__wrapped__(self, *args, **kwargs)

    def _write_agent_error_flags(self) -> None:
        super()._write_agent_error_flags()
        failure = self.logs_dir / "llm-call-failure.json"
        if failure.exists():
            path = self.logs_dir / "agent_error_flags.json"
            flags = json.loads(path.read_text()) if path.exists() else {}
            flags["llm_call_failure"] = json.loads(failure.read_text())
            path.write_text(json.dumps(flags))
