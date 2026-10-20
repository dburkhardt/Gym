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

from pydantic import Field

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
    uuid: str
    verifier_metadata: dict


class VerifyResponse(BaseVerifyResponse, VerifyRequest):
    scoring_details: dict


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
        details = await run_score(
            benchmark=self.config.benchmark,
            upstream_repo=Path(self.config.upstream_repo),
            task={"uuid": body.uuid, "verifier_metadata": body.verifier_metadata},
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
        return VerifyResponse(
            **body.model_dump(by_alias=True), reward=details["reward"], scoring_details=details, **accounting
        )


if __name__ == "__main__":
    CreatorResourcesServer.run_webserver()
