# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native creator MOOSE-Chem stages, collected as a Gym agent trajectory."""

import asyncio
import hashlib
import json
import secrets
import shlex
import time
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import Request, Response
from pydantic import ConfigDict, Field, PrivateAttr

from benchmarks.moosechem.native import MODEL, append_durable
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict, get_global_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import current_rollout_id, maybe_rollout_id_from_run_body
from nemo_gym.sandbox.model_bridge import bridge
from nemo_gym.sandbox.providers.local.provider import LocalProvider
from nemo_gym.sandbox.scoring_workspace import scoring_workspace
from nemo_gym.server_utils import rollout_path_prefix


class Config(BaseResponsesAPIAgentConfig):
    native_python: Path
    upstream_repo: Path
    output_root: Path
    frozen_tasks: Path
    import_existing: bool = False
    model_server: ModelServerRef = Field(
        default_factory=lambda: ModelServerRef(type="responses_api_models", name="policy_model")
    )
    blocked_task_ids: list[int] = Field(default_factory=list)
    output_token_budget: Literal[8192, 16384] = 8192
    native_timeout_seconds: float = Field(default=14400, gt=0)


class RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict


class VerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    artifact_dir: str
    native_metrics: dict
    reward_metric: str = "final_round_inspiration_top3_recall_not_hypothesis_quality"
    synthetic_summary_response: bool = True


class Agent(SimpleResponsesAPIAgent):
    config: Config
    _semaphore: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(1))

    async def responses(
        self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        raise NotImplementedError("Use /run with frozen task identifier")

    def execute(self, task_id):
        out = self.config.output_root / f"task{task_id}"
        if out.exists():
            if not self.config.import_existing or not (out / "result.json").exists():
                raise ValueError("Existing native attempt must be reconciled, not rerun")
        else:
            raise ValueError("Native execution must use the managed asynchronous path")
        return Agent.read_result(self, task_id)

    def read_result(self, task_id):
        out = self.config.output_root / f"task{task_id}"
        result = json.loads((out / "result.json").read_text())
        screen = json.loads((out / "screening.json").read_text())
        ratios = list(screen[1].values())
        assert len(ratios) == 1 and len(ratios[0]) == 4
        reward = ratios[0][-1][1]
        assert 0 <= reward <= 1
        traces = [json.loads(x) for x in (out / "model-traces.jsonl").read_text().splitlines()]
        assert traces and all(t["request"]["model"] == MODEL for t in traces)
        assert all(t.get("response", {}).get("model") == MODEL for t in traces if "response" in t)
        return out, result, screen, reward, len(traces)

    async def execute_native(self, task_id: int, body: RunRequest) -> None:
        if task_id in self.config.blocked_task_ids:
            raise ValueError("Task has an unresolved historical attempt; reconcile before execution")
        global_config = get_global_config_dict()
        if not global_config.get("provider_dispatch_guard") or global_config.get("policy_model_name") != MODEL:
            raise ValueError("MOOSE-Chem requires guarded exact-model inference")
        capture_id = current_rollout_id() or maybe_rollout_id_from_run_body(body)
        if capture_id is None:
            raise ValueError("Native MOOSE-Chem requires a rollout identity")
        server = get_first_server_config_dict(global_config, self.config.model_server.name)
        endpoint = f"http://{server['host']}:{server['port']}" + rollout_path_prefix(capture_id) + "/v1"
        out = self.config.output_root / f"task{task_id}"
        out.mkdir(parents=True, exist_ok=False)
        source = Path(__file__).resolve().parents[2]
        lineage = uuid4().hex
        files = [source / "benchmarks/moosechem" / name for name in ("native.py", "guarded.py", "source-sha256.json")]
        files.append(Path(__file__).resolve())
        interpreter = self.config.native_python.resolve()
        identity = {
            "task": task_id,
            "model": MODEL,
            "budget": self.config.output_token_budget,
            "timeout_seconds": self.config.native_timeout_seconds,
            "source": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
            "interpreter": {"path": str(interpreter), "sha256": hashlib.sha256(interpreter.read_bytes()).hexdigest()},
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        append_durable(out / "execution-identity.jsonl", {"config_sha256": digest, **identity})
        provider = LocalProvider(concurrency=1, workspace_root="/tmp")
        async with scoring_workspace(provider) as handle:
            socket = handle.raw["workspace"] / "model.sock"
            token = secrets.token_urlsafe(32)
            async with bridge(
                socket,
                endpoint=endpoint,
                token=token,
                lineage=lineage,
                digest=digest,
                namespace="moosechem",
                expected_model=MODEL,
            ):
                command = shlex.join(
                    [
                        str(self.config.native_python),
                        "-m",
                        "benchmarks.moosechem.guarded",
                        "--repo",
                        str(self.config.upstream_repo),
                        "--output",
                        str(out),
                        "--task-id",
                        str(task_id),
                        "--output-token-budget",
                        str(self.config.output_token_budget),
                    ]
                )
                result = await provider.exec(
                    handle,
                    command,
                    timeout_s=self.config.native_timeout_seconds,
                    env={
                        "PYTHONPATH": str(source),
                        "NEMO_GYM_EXTRA_ROOTS": str(source),
                        "MOOSE_SOCKET": str(socket),
                        "MOOSE_EXECUTION_IDENTITY": str(out / "execution-identity.jsonl"),
                        "MOOSE_SOCKET_TOKEN": token,
                        "MOOSE_LINEAGE": lineage,
                        "MOOSE_CONFIG_SHA256": digest,
                    },
                )
                (out / "stdout.log").write_text(result.stdout or "")
                (out / "stderr.log").write_text(result.stderr or "")
                if result.return_code != 0 or result.error_type:
                    raise RuntimeError("Native MOOSE-Chem failed; preserved failure is not a scientific zero")

    async def run(self, request: Request, body: RunRequest) -> VerifyResponse:
        task_id = body.verifier_metadata["id"]
        tasks = {r["id"]: r for r in map(json.loads, self.config.frozen_tasks.read_text().splitlines())}
        source = tasks[task_id]
        expected = source["question"] + "\n\n" + source["survey"]
        actual = body.responses_create_params.model_dump()["input"]
        assert len(actual) == 1 and actual[0]["role"] == "user" and actual[0]["content"] == expected
        async with self._semaphore:
            if not self.config.import_existing:
                await self.execute_native(task_id, body)
            # Loading a completed managed attempt does not dispatch inference.
            out, result, screen, reward, n = self.read_result(task_id)
        text = json.dumps({"scope": result["scope"], "native_evaluation": result["evaluation"]})
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
            artifact_dir=str(out),
            native_metrics={
                "retrieval_by_round": screen[1],
                "full_creator_evaluation": result["evaluation"],
                "actual_model_calls": n,
                "judge_variant": "Nemotron3.5SuperVL_all_roles",
                "imported_saved_execution": self.config.import_existing,
            },
        )


if __name__ == "__main__":
    Agent.run_webserver()
