# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native MetaSyn BM25/RAG generator and complete upstream evaluator through Gym."""

import asyncio
import hashlib
import json
import math
import secrets
import shutil
import time
from pathlib import Path
from uuid import uuid4

from fastapi import Request, Response
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict, get_global_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import current_rollout_id, maybe_rollout_id_from_run_body
from nemo_gym.sandbox.lifecycle import _publish
from nemo_gym.sandbox.model_bridge import bridge
from nemo_gym.sandbox.providers.local.provider import LocalProvider
from nemo_gym.sandbox.scoring_workspace import scoring_workspace
from nemo_gym.server_utils import rollout_path_prefix
from responses_api_agents.metasyn_native.artifacts import assert_prior_attempts_settled, attempt_root
from responses_api_agents.metasyn_native.execution import run_creator
from responses_api_agents.metasyn_native.identity import creator_identity, scientific_assets
from responses_api_agents.metasyn_native.trajectory import observations


MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


def native_reward(evaluation: dict) -> float:
    if evaluation.get("status") != "succeeded":
        raise ValueError("Native grading did not succeed; no scientific reward is assigned")
    metric = evaluation.get("metrics", {}).get("inclusion_f1")
    if isinstance(metric, bool) or not isinstance(metric, (int, float)):
        raise ValueError("Missing or nonnumeric native inclusion_f1; no scientific reward is assigned")
    reward = float(metric)
    if not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError("Native inclusion_f1 is nonfinite or out of range")
    return reward


class Config(BaseResponsesAPIAgentConfig):
    native_python: str
    runner_script: Path
    materialized_root: Path
    output_root: Path
    reference_file: Path
    creator_source_root: Path = Path("/unqualified-metasyn-source")
    scientific_threads: int = Field(default=1, ge=1)
    inference_enabled: bool = False
    model_server: ModelServerRef = Field(
        default_factory=lambda: ModelServerRef(type="responses_api_models", name="policy_model")
    )


class RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, int]


class VerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    metrics: dict
    evaluation_status: str
    artifact_dir: str
    reward_metric: str = "inclusion_f1"
    ng_trajectory: dict


class MetaSynAgent(SimpleResponsesAPIAgent):
    config: Config
    _semaphore: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(1))

    async def responses(
        self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        raise NotImplementedError("Use /run with a prepared review identifier")

    async def _evaluate(self, review_id: int, endpoint: str, lineage: str):
        root = attempt_root(self.config.output_root, review_id, lineage)
        root.mkdir(parents=True, exist_ok=False)
        out = root / str(review_id)
        out.mkdir()
        source = Path(__file__).resolve().parents[2]
        files = [
            Path(__file__),
            self.config.runner_script,
            Path(__file__).with_name("execution.py"),
            Path(__file__).with_name("guarded_clients.py"),
            Path(__file__).with_name("identity.py"),
            Path(__file__).with_name("trajectory.py"),
            Path(__file__).with_name("artifacts.py"),
        ]
        interpreter = Path(shutil.which(self.config.native_python) or self.config.native_python).resolve()
        identity = {
            "scientific_assets": scientific_assets(self.config.materialized_root, review_id),
            "creator": creator_identity(self.config.creator_source_root),
            "interpreter": {"path": str(interpreter), "sha256": hashlib.sha256(interpreter.read_bytes()).hexdigest()},
            "review_id": review_id,
            "model": MODEL,
            "lineage": lineage,
            "prompt_sha256": hashlib.sha256(
                (self.config.materialized_root / "rag" / str(review_id) / "prompt.txt").read_bytes()
            ).hexdigest(),
            "reference_path": str(self.config.reference_file.resolve()),
            "reference_sha256": hashlib.sha256(self.config.reference_file.read_bytes()).hexdigest(),
            "sources": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "timeout_s": 3600,
            "native_python": self.config.native_python,
        }
        threads = getattr(self.config, "scientific_threads", 1)
        identity["scientific_threads"] = threads
        global_config = get_global_config_dict()
        if global_config.get("model_call_capture_dir"):
            identity["capture_root"] = str(Path(global_config["model_call_capture_dir"]).resolve())
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        _publish(out / "identity.json", identity)
        provider = LocalProvider(concurrency=1)
        async with scoring_workspace(provider) as handle:
            socket = handle.raw["workspace"] / "model.sock"
            token = secrets.token_urlsafe(32)
            async with bridge(
                socket,
                endpoint=endpoint,
                token=token,
                lineage=lineage,
                digest=digest,
                namespace="metasyn",
                expected_model=MODEL,
            ):
                await run_creator(
                    [
                        self.config.native_python,
                        str(self.config.runner_script),
                        "--review-id",
                        str(review_id),
                        "--output-root",
                        str(root),
                        "--reference-file",
                        str(self.config.reference_file),
                    ],
                    out,
                    timeout_s=3600,
                    env={
                        "PYTHONPATH": str(source) + ":" + str(self.config.creator_source_root.parent),
                        "NEMO_GYM_EXTRA_ROOTS": str(source),
                        "METASYN_DATA": str(self.config.materialized_root),
                        "OMP_NUM_THREADS": str(threads),
                        "OPENBLAS_NUM_THREADS": str(threads),
                        "TOKENIZERS_PARALLELISM": "false",
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "METASYN_BRIDGE_SOCKET": str(socket),
                        "METASYN_BRIDGE_TOKEN": token,
                        "METASYN_BRIDGE_LINEAGE": lineage,
                        "METASYN_BRIDGE_DIGEST": digest,
                    },
                )
        evaluation = json.loads((out / "evaluation.json").read_text())
        return out, evaluation

    async def run(self, request: Request, body: RunRequest) -> VerifyResponse:
        if not self.config.inference_enabled:
            raise ValueError("MetaSyn execution requires explicit qualified-runtime enablement")
        global_config = get_global_config_dict()
        if not global_config.get("provider_dispatch_guard") or global_config.get("policy_model_name") != MODEL:
            raise ValueError("MetaSyn requires the guarded authorized model")
        lineage = current_rollout_id() or maybe_rollout_id_from_run_body(body)
        if not lineage:
            raise ValueError("Native rollout identity is required")
        server = get_first_server_config_dict(global_config, self.config.model_server.name)
        endpoint = f"http://{server['host']}:{server['port']}" + rollout_path_prefix(lineage) + "/v1"
        review_id = body.verifier_metadata["review_id"]
        supplied = body.responses_create_params.model_dump()["input"]
        expected = (self.config.materialized_root / "rag" / str(review_id) / "prompt.txt").read_text()
        if len(supplied) != 1 or supplied[0].get("role") != "user" or supplied[0].get("content") != expected:
            raise ValueError("Gym input differs from the preserved native prompt")
        async with self._semaphore:
            capture_root = global_config.get("model_call_capture_dir")
            if not capture_root:
                raise ValueError("MetaSyn requires its durable model capture store")
            assert_prior_attempts_settled(self.config.output_root, review_id, lineage, Path(capture_root))
            out, evaluation = await self._evaluate(review_id, endpoint, lineage)
        report = (out / "report.md").read_text()
        trajectory, usage = observations(
            out / "provider-calls",
            task_id=str(body.model_dump().get("task_id", f"metasyn::{review_id}")),
            rollout_id=lineage,
            model_ref=self.config.model_server,
        )
        response = NeMoGymResponse.model_validate(
            {
                "id": "resp_" + uuid4().hex,
                "object": "response",
                "created_at": time.time(),
                "model": MODEL,
                "status": "completed",
                "usage": usage,
                "output": [
                    {
                        "id": "msg_" + uuid4().hex,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "annotations": [], "text": report}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        # Gym needs one scalar. Preserve the named native component and every other metric;
        # this is not an invented overall benchmark score or a pass/fail claim.
        reward = native_reward(evaluation)
        return VerifyResponse(
            **body.model_dump(),
            response=response,
            reward=reward,
            metrics=evaluation["metrics"],
            evaluation_status=evaluation["status"],
            artifact_dir=str(out),
            ng_trajectory=trajectory.model_dump(mode="json"),
        )


if __name__ == "__main__":
    MetaSynAgent.run_webserver()
