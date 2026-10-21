# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Maintained native Verifiers/OpenCode execution; saved importer is separate."""

import asyncio
import hashlib
import json
import os
import shlex
import time
from pathlib import Path
from uuid import uuid4

from fastapi import Request, Response
from pydantic import ConfigDict, Field, PrivateAttr

from benchmarks.bioagentbench.outcomes import disposition
from benchmarks.bioagentbench.prepare import rows
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict, get_global_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import current_rollout_id, maybe_rollout_id_from_run_body
from nemo_gym.sandbox.guarded_chat_transport import publish
from nemo_gym.sandbox.providers.local.provider import LocalProvider
from nemo_gym.sandbox.scoring_workspace import scoring_workspace
from nemo_gym.server_utils import rollout_path_prefix


MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


def selected_interpreter(path: Path) -> tuple[Path, dict]:
    """Keep venv executable spelling; resolve only the underlying binary hash."""
    executable = Path(os.path.abspath(path))
    return executable, {
        "path": str(executable),
        "binary_path": str(executable.resolve()),
        "sha256": hashlib.sha256(executable.resolve().read_bytes()).hexdigest(),
    }


class Config(BaseResponsesAPIAgentConfig):
    native_python: Path
    creator_root: Path
    metadata_path: Path
    asset_root: Path
    asset_manifest: Path
    image: str
    output_root: Path
    blocked_task_ids: list[str] = Field(default_factory=list)
    model_server: ModelServerRef = Field(
        default_factory=lambda: ModelServerRef(type="responses_api_models", name="policy_model")
    )


class RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: dict


class VerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    native_metrics: dict
    artifact_dir: str
    imported_saved_execution: bool = False
    synthetic_summary_response: bool = True


class Agent(SimpleResponsesAPIAgent):
    config: Config
    _semaphore: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(1))

    async def responses(
        self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        raise NotImplementedError("Use /run with a frozen workflow")

    async def run(self, request: Request, body: RunRequest) -> VerifyResponse:
        task_id = body.verifier_metadata["id"]
        frozen = next((row for row in rows(self.config.metadata_path) if row["task_id"] == task_id), None)
        if (
            frozen is None
            or body.verifier_metadata != frozen["verifier_metadata"]
            or body.responses_create_params.model_dump()["input"] != frozen["responses_create_params"]["input"]
        ):
            raise ValueError("Frozen task prompt or metadata changed")
        if task_id in self.config.blocked_task_ids:
            raise ValueError("Historical task requires reconciliation before a new attempt")
        global_config = get_global_config_dict()
        if not global_config.get("provider_dispatch_guard") or global_config.get("policy_model_name") != MODEL:
            raise ValueError("Guarded exact-model inference required")
        capture = current_rollout_id() or maybe_rollout_id_from_run_body(body)
        if capture is None:
            raise ValueError("Stable Gym rollout identity required")
        server = get_first_server_config_dict(global_config, self.config.model_server.name)
        endpoint = f"http://{server['host']}:{server['port']}" + rollout_path_prefix(capture) + "/v1"
        claim = hashlib.sha256((str(capture) + ":" + task_id).encode()).hexdigest()
        out = self.config.output_root / claim
        out.mkdir(parents=True, exist_ok=False)
        root = Path(__file__).resolve().parents[2]
        files = [
            Path(__file__).resolve(),
            *(root / "benchmarks/bioagentbench").glob("*.py"),
            root / "benchmarks/bioagentbench/source-sha256.json",
            root / "nemo_gym/sandbox/api.py",
            root / "nemo_gym/sandbox/lifecycle.py",
        ]
        files += list((root / "nemo_gym/sandbox").rglob("*.py"))
        files += [root / "nemo_gym" / name for name in ("server_utils.py", "provider_dispatch.py", "openai_utils.py")]
        interpreter, interpreter_receipt = selected_interpreter(self.config.native_python)
        native = {
            "model": MODEL,
            "num_rollouts": 1,
            "retries": {"rollout": {"max_retries": 0}},
            "max_concurrent": 1,
            "push": False,
            "rich": False,
            "sampling": {"temperature": 0.2, "reasoning_effort": "high", "max_tokens": 32768},
            "client": {"type": "eval", "base_url": "http://native/v1", "api_key_var": "BIOAGENT_LOCAL_KEY"},
            "taskset": {
                "id": "bioagent-taskset",
                "metadata_path": str(self.config.metadata_path),
                "data_root": str(self.config.asset_root),
                "task_id": task_id,
                "include_reference": True,
                "staging": "local",
                "task": {
                    "artifact_root": str(out / "artifacts"),
                    "judge": {
                        "model": MODEL,
                        "base_url": "http://native/v1",
                        "api_key_var": "BIOAGENT_LOCAL_KEY",
                        "sampling": {"reasoning_effort": "medium"},
                    },
                },
            },
            "harness": {
                "id": "bioagent-opencode",
                "version": "1.15.11",
                "runtime": {"type": "docker", "image": self.config.image, "cpu": 2, "memory": 8},
            },
            "timeout": {"setup": 3600, "rollout": 14400, "finalize": 3600, "scoring": 3600},
        }
        packet = {
            "task_id": task_id,
            "model": MODEL,
            "image": self.config.image,
            "endpoint": endpoint,
            "lineage": str(capture),
            "creator_root": str(self.config.creator_root),
            "asset_manifest": str(self.config.asset_manifest),
            "asset_manifest_sha256": hashlib.sha256(self.config.asset_manifest.read_bytes()).hexdigest(),
            "native_config": native,
            "selected_sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
            "interpreter": interpreter_receipt,
        }
        packet["digest"] = hashlib.sha256(json.dumps(packet, sort_keys=True).encode()).hexdigest()
        publish(out / "packet.json", packet)
        async with self._semaphore:
            provider = LocalProvider(concurrency=1, workspace_root="/tmp")
            async with scoring_workspace(provider) as handle:
                result = await provider.exec(
                    handle,
                    shlex.join(
                        [
                            str(interpreter),
                            "-m",
                            "benchmarks.bioagentbench.worker",
                            "--packet",
                            str(out / "packet.json"),
                        ]
                    ),
                    timeout_s=25980,
                    terminate_background_children=True,
                    env={
                        "PYTHONPATH": str(root),
                        "NEMO_GYM_EXTRA_ROOTS": str(root),
                        "BIOAGENT_LOCAL_KEY": "task-local-placeholder",
                    },
                )
                (out / "stdout.log").write_text(result.stdout or "")
                (out / "stderr.log").write_text(result.stderr or "")
                if result.error_type or result.return_code:
                    raise RuntimeError("Native BioAgent execution failed; not a scientific zero")
        traces = json.loads((out / "result.json").read_text())["traces"]
        if len(traces) != 1 or traces[0]["task"]["data"]["task_id"] != task_id:
            raise ValueError("Native task result mismatch")
        trace = traces[0]
        outcome = disposition(trace)
        publish(out / "disposition.json", outcome)
        if not outcome["measurement_available"]:
            raise RuntimeError("Native attempted trajectory has no scientific score: " + outcome["disposition"])
        reward = outcome["scientific_reward"]
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
                        "content": [
                            {
                                "type": "output_text",
                                "annotations": [],
                                "text": json.dumps(
                                    {
                                        "native_metrics": trace["metrics"],
                                        "native_assessment": trace["info"].get("assessment"),
                                    }
                                ),
                            }
                        ],
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
            artifact_dir=str(out),
            native_metrics={
                "metrics": trace["metrics"],
                "rewards": trace["rewards"],
                "protocol": "creator_Verifiers_OpenCode_managed_runtime",
                "disposition": outcome,
            },
        )


if __name__ == "__main__":
    Agent.run_webserver()
