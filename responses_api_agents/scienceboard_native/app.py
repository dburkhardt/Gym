# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native ScienceBoard Tester in an owned worker; image qualification is mandatory."""

import asyncio
import hashlib
import json
import secrets
import shlex
import time
from pathlib import Path
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict, Field, PrivateAttr

from benchmarks.scienceboard.admission import admit
from benchmarks.scienceboard.guarded_model import PROTOCOL_VARIANT as PROTOCOL
from benchmarks.scienceboard.prepare import MANIFEST, checked_source
from benchmarks.scienceboard.qualification import (
    implementation_identity,
    settle_cleanup,
    validate_receipt,
)
from benchmarks.scienceboard.worker import MODEL
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict, get_global_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import current_rollout_id, maybe_rollout_id_from_run_body
from nemo_gym.sandbox.guarded_chat_transport import publish
from nemo_gym.sandbox.model_bridge import bridge
from nemo_gym.sandbox.providers.local.provider import LocalProvider
from nemo_gym.sandbox.scoring_workspace import scoring_workspace
from nemo_gym.server_utils import rollout_path_prefix


class Config(BaseResponsesAPIAgentConfig):
    native_python: Path
    creator_source: Path
    output_root: Path
    image_inspect: Path
    image_audit: Path
    placement: Path
    runtime_qualified: bool = False
    runtime_receipt: Path | None = None
    timeout_seconds: float = Field(default=7200, gt=0)
    model_server: ModelServerRef


class RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict


class Result(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    artifact_dir: str


def map_result(body: RunRequest, record: dict, out: Path) -> Result:
    """Unscored model attempts remain records, without numeric placeholder aggregation."""
    raw = record["raw_reward"]
    metadata = dict(getattr(body, "metadata", None) or {})
    metadata.update(
        scienceboard=record,
        verifier_result={"rewards": {} if raw is None else {"reward": raw}},
        measurement_review=record["measurement_review"],
    )
    response = NeMoGymResponse.model_validate(
        dict(
            id=out.name,
            created_at=time.time(),
            model=MODEL,
            object="response",
            output=[],
            tools=[],
            tool_choice="auto",
            parallel_tool_calls=False,
        )
    )
    values = body.model_dump()
    values.update(
        response=response,
        reward=raw if raw is not None else 0.0,
        mask_sample=raw is None or record["mask_sample"],
        metadata=metadata,
        artifact_dir=str(out),
        failure_kind=(
            "scienceboard:" + record["transport_outcome"]
            if record["transport_outcome"] in {"transport_unresolved", "provider_content_filter"}
            else "model_budget"
            if record["disposition"] == "model_output_budget"
            else "scienceboard:model_invalid_action"
            if record["disposition"] == "model_invalid_action"
            else "verifier_error"
            if record["disposition"] == "verifier_artifact_failure"
            else "agent_run_error"
            if raw is None or record["mask_sample"]
            else None
        ),
    )
    return Result.model_validate(values)


class Agent(SimpleResponsesAPIAgent):
    config: Config
    _sem: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(1))

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        raise NotImplementedError("Use /run with a prepared ScienceBoard task")

    async def run(self, body: RunRequest, request: Request = None) -> Result:
        async with self._sem:
            return await self._run(body)

    async def _run(self, body: RunRequest) -> Result:
        cfg = self.config
        manifest = json.loads(MANIFEST.read_text())
        item = next(x for x in manifest["tasks"] if x["task_id"] == body.verifier_metadata["task_id"])
        expected = dict(
            task_id=item["task_id"],
            task_sha256=item["sha256"],
            creator_task_path=item["path"],
            source_revision=manifest["source_revision"],
        )
        if body.verifier_metadata != expected:
            raise ValueError("Frozen task metadata changed")
        task = json.loads(checked_source(cfg.creator_source, item["path"], item["sha256"]).read_text())
        messages = body.responses_create_params.model_dump(exclude_none=True)["input"]
        if len(messages) != 1:
            raise ValueError("Frozen instruction changed")
        message = dict(messages[0])
        if message.get("type") == "message":
            message.pop("type")
        if message != {"role": "user", "content": task["instruction"]}:
            raise ValueError("Frozen instruction changed")
        for name, sha in manifest["files"].items():
            checked_source(cfg.creator_source, name, sha)
        image, audit, resources = (
            json.loads(p.read_text()) for p in (cfg.image_inspect, cfg.image_audit, cfg.placement)
        )
        admission = admit(image, audit, resources)
        if admission["blockers"]:
            raise ValueError(f"ScienceBoard image admission blocked: {admission['blockers']}")
        if not cfg.runtime_qualified:
            raise ValueError(
                "ScienceBoard runtime unqualified: desktop initialization and hard-death cleanup required"
            )
        if cfg.runtime_receipt is None:
            raise ValueError("Runtime qualification requires an image/source-bound evidence receipt")
        receipt = json.loads(cfg.runtime_receipt.read_text())
        required = dict(
            image_id=image["Id"],
            source_revision=manifest["source_revision"],
            selection_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            protocol=PROTOCOL,
            task_id=item["task_id"],
            desktop_initialized=True,
            controller_death_cleanup=True,
        )
        implementation = implementation_identity(cfg.native_python)
        expected_runtime = validate_receipt(receipt, implementation=implementation, required=required)
        global_config = get_global_config_dict()
        if not global_config.get("provider_dispatch_guard") or global_config.get("policy_model_name") != MODEL:
            raise ValueError("ScienceBoard requires guarded exact-model inference")
        capture = current_rollout_id() or maybe_rollout_id_from_run_body(body)
        if capture is None:
            raise ValueError("Native rollout identity required")
        server = get_first_server_config_dict(global_config, cfg.model_server.name)
        endpoint = f"http://{server['host']}:{server['port']}" + rollout_path_prefix(capture) + "/v1"
        source = Path(__file__).resolve().parents[2]
        out = cfg.output_root.absolute() / uuid4().hex
        out.mkdir(parents=True, exist_ok=False)
        python = cfg.native_python.absolute()  # Do not dereference a venv launcher.
        identity = dict(
            task=expected,
            protocol=PROTOCOL,
            image=image["Id"],
            timeout=cfg.timeout_seconds,
            admission=admission,
            **implementation,
            runtime=expected_runtime,
        )
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        publish(out / "identity.json", identity)
        provider = LocalProvider(concurrency=1, workspace_root="/tmp")
        async with scoring_workspace(provider) as handle:
            socket = handle.raw["workspace"] / "model.sock"
            token, lineage = secrets.token_urlsafe(32), uuid4().hex
            packet = dict(
                source=str(cfg.creator_source.absolute()),
                output=str(out),
                task_id=item["task_id"],
                image=image,
                audit=audit,
                resources=resources,
                socket=str(socket),
                lineage=lineage,
                digest=digest,
                protocol=PROTOCOL,
                implementation=identity["modules"],
                expected_runtime=expected_runtime,
            )
            publish(out / "packet.json", packet)
            async with bridge(
                socket,
                endpoint=endpoint,
                token=token,
                lineage=lineage,
                digest=digest,
                namespace="scienceboard",
                expected_model=MODEL,
            ):
                try:
                    result = await provider.exec(
                        handle,
                        shlex.join(
                            [str(python), "-m", "benchmarks.scienceboard.worker", "--packet", str(out / "packet.json")]
                        ),
                        timeout_s=cfg.timeout_seconds,
                        terminate_background_children=True,
                        env={
                            "PYTHONPATH": str(source) + ":" + str(cfg.creator_source.absolute()),
                            "NEMO_GYM_EXTRA_ROOTS": str(source),
                            "SCIENCEBOARD_BRIDGE_TOKEN": token,
                        },
                    )
                    (out / "stdout.log").write_text(result.stdout or "")
                    (out / "stderr.log").write_text(result.stderr or "")
                    if result.return_code != 0 or result.error_type:
                        raise RuntimeError(
                            "Owned creator worker failed; preserve artifacts and reconcile before retry"
                        )
                finally:
                    await settle_cleanup(out / "desktops")
        return map_result(body, json.loads((out / "result.json").read_text()), out)


if __name__ == "__main__":
    Agent.run_webserver()
