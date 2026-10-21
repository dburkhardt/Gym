# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Gym verifier delegating to pinned creator scoring in isolated processes."""

import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict, Field

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
    benchmark: str
    upstream_repo: str
    scoring_timeout_s: float = Field(default=600, gt=0, allow_inf_nan=False)


class VerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    uuid: str
    verifier_metadata: dict


class VerifyResponse(BaseVerifyResponse, VerifyRequest):
    model_config = ConfigDict(extra="allow")
    scoring_details: dict
    metadata: dict = Field(default_factory=dict)


async def run_score(*, benchmark: str, upstream_repo: Path, task: dict, answer: str, timeout_s: float) -> dict:
    """Run trusted creator code in a cancellable process group, not an executor pool.

    This is host execution, not a security sandbox. LocalProvider kills ordinary
    descendants on timeout/cancellation; detached sessions and SIGKILL of the
    resource server require an external process supervisor.
    """
    provider = LocalProvider(concurrency=2)
    async with scoring_workspace(provider) as handle:
        workspace = handle.raw["workspace"]
        request_path = workspace / "request.json"
        result_path = workspace / "result.json"
        request_path.write_text(
            json.dumps({"benchmark": benchmark, "repo": str(upstream_repo.resolve()), "task": task, "answer": answer})
        )
        source_root = Path(__file__).resolve().parents[2]
        worker = Path(__file__).with_name("score_worker.py").resolve()
        command = shlex.join([sys.executable, str(worker), str(request_path), str(result_path)])
        # Keep the selected source first, irrespective of editable install precedence.
        env = {"PYTHONPATH": os.pathsep.join([str(source_root), *sys.path])}
        result = await provider.exec(handle, command, env=env, timeout_s=timeout_s)
        if result.error_type == "timeout":
            raise TimeoutError(f"Creator scorer {benchmark} exceeded {timeout_s:g}s; no score produced")
        if result.return_code != 0:
            raise RuntimeError(f"Creator scorer {benchmark} failed (exit {result.return_code}): {result.stderr}")
        envelope = json.loads(result_path.read_text())
        source = source_root / "benchmarks/scientific_agent_campaign/deterministic.py"
        expected = {"source_path": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        if envelope.get("runtime") != expected:
            raise RuntimeError("Creator scorer imported a different source than the selected runtime")
        details = envelope["details"]
        if not isinstance(details, dict):
            raise TypeError("Creator scorer returned a non-object result")
        return details


class CreatorResourcesServer(SimpleResourcesServer):
    config: Config
    ray_enabled = False

    async def verify(self, body: VerifyRequest) -> VerifyResponse:
        task = {"uuid": body.uuid, "verifier_metadata": body.verifier_metadata}
        if self.config.benchmark == "scienceqa":
            task["responses_create_params"] = body.responses_create_params.model_dump(mode="json")
        details = await run_score(
            benchmark=self.config.benchmark,
            upstream_repo=Path(self.config.upstream_repo),
            task=task,
            answer=body.response.output_text or "",
            timeout_s=self.config.scoring_timeout_s,
        )
        if details.get("reward") is None:
            raise ValueError("This task has no upstream automatic score; do not convert to zero.")
        accounting = {}
        if body.response.status == "incomplete":
            reason = getattr(body.response.incomplete_details, "reason", None)
            accounting = {
                "failure_kind": "model_budget" if reason == "max_output_tokens" else "incomplete_response",
                "failure_reason": "Model response incomplete: " + str(reason),
            }
        metadata = {}
        if self.config.benchmark == "qcbench":
            evidence = details.get("measurement_evidence")
            task = {"uuid": body.uuid, "verifier_metadata": body.verifier_metadata}
            task_sha = hashlib.sha256(json.dumps(task, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if not isinstance(evidence, dict) or evidence.get("task_sha256") != task_sha:
                raise ValueError("QCBench score lacks matching task/reference evidence")
            if not isinstance(details.get("measurement_valid"), bool):
                raise ValueError("QCBench score lacks explicit reference measurement validity")
            invalid = details["measurement_valid"] is False
            if invalid:
                accounting = {
                    "mask_sample": True,
                    "failure_kind": "qcbench:invalid_reference_measurement",
                    "failure_reason": details["measurement_invalid_reason"],
                }
            metadata = {
                "source": "qcbench",
                "verifier_result": {"rewards": {"creator_numeric": details["reward"]}},
                "measurement_review": {
                    "version": "qcbench-reference-self-consistency-v1",
                    "implementation": {
                        "import_path": "resources_servers.campaign_creator_scoring.app:CreatorResourcesServer.verify",
                        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    },
                    "evidence": evidence,
                    "metrics": {
                        "creator_numeric": {
                            "status": "invalid" if invalid else "unresolved",
                            "reason": details.get("measurement_invalid_reason")
                            or "Reference self-consistency passed; scientific reference correctness has not been reviewed",
                        }
                    },
                },
            }
        if self.config.benchmark == "scienceqa":
            review = details["measurement_review"]
            expected = hashlib.sha256(json.dumps(task, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if review["evidence"]["task_sha256"] != expected:
                raise ValueError("ScienceQA review does not match scored input")
            metadata = {
                "source": "scienceqa",
                "verifier_result": {
                    "rewards": {
                        "creator_seeded_fallback_accuracy": details["reward"],
                        "strict_accuracy": details["strict_score"],
                    }
                },
                "measurement_review": review,
            }
            if any(metric["status"] == "invalid" for metric in review["metrics"].values()):
                accounting.update(
                    mask_sample=True,
                    failure_kind="scienceqa:invalid_measurement",
                    failure_reason="See separate strict and seeded-fallback metric qualifications",
                )
        if self.config.benchmark == "fgbench":
            from benchmarks.fgbench.projection import project

            metadata.update(
                project(
                    Path(self.config.upstream_repo),
                    task,
                    details,
                    protocol={
                        "model": body.response.model,
                        "temperature": body.responses_create_params.temperature,
                        "max_output_tokens": body.responses_create_params.max_output_tokens,
                    },
                )
            )
        payload = body.model_dump(by_alias=True)
        # Preserve native task/rollout lineage, but never retain a prior regrade's
        # reward, mask or measurement review as if it were this verification.
        previous_metadata = payload.get("metadata")
        merged_metadata = dict(previous_metadata) if isinstance(previous_metadata, dict) else {}
        for key in ("measurement_review", "verifier_result"):
            merged_metadata.pop(key, None)
        merged_metadata.update(metadata)
        payload.update(
            reward=details["reward"],
            scoring_details=details,
            metadata=merged_metadata,
            mask_sample=False,
            failure_kind=None,
            failure_reason=None,
        )
        payload.update(accounting)
        return VerifyResponse(**payload)


if __name__ == "__main__":
    CreatorResourcesServer.run_webserver()
