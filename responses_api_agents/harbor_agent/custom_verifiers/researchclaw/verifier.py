# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harbor scientific verifier for the pinned ResearchClaw creator checklist."""

import asyncio
import hashlib
import json
import os
import signal
import uuid
from pathlib import Path

from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from nemo_gym.global_config import get_global_config_dict
from responses_api_agents.harbor_agent.custom_verifiers.rewardkit_agent.verifier import _publish, _settle


class ResearchClawVerifier(BaseVerifier):
    def __init__(
        self,
        *,
        source_root: str,
        frozen_manifest: str,
        frozen_manifest_sha256: str,
        gym_model_api_base: str,
        gym_model_name: str,
        controller_python: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if gym_model_name != "nvidia/nvidia/nemotron-3.5-super-vl":
            raise ValueError("ResearchClaw campaign requires exact Nemotron model")
        if "/ng-rollout/" not in gym_model_api_base or not gym_model_api_base.endswith("/v1"):
            raise ValueError("ResearchClaw requires guarded Gym capture endpoint")
        if not get_global_config_dict().get("provider_dispatch_guard"):
            raise ValueError("ResearchClaw requires provider_dispatch_guard enabled")
        self.controller_python = controller_python
        self.source = Path(source_root).resolve()
        self.manifest_path = Path(frozen_manifest).resolve()
        if hashlib.sha256(self.manifest_path.read_bytes()).hexdigest() != frozen_manifest_sha256:
            raise ValueError("Frozen task manifest hash mismatch")
        self.manifest = json.loads(self.manifest_path.read_text())
        self.endpoint = gym_model_api_base

    async def verify(self):
        task_id = self.task.paths.task_dir.name
        if task_id not in self.manifest["tasks"]:
            raise ValueError("Task is absent from frozen ResearchClaw manifest")
        out = self.trial_paths.verifier_dir / "creator-scoring"
        # Explicit fresh verifier attempt. Never reset an unresolved prior controller.
        out.mkdir(parents=True, exist_ok=False)
        lineage = uuid.uuid4().hex
        _publish(
            out / "identity.json",
            {"lineage": lineage, "task_id": task_id, "recovery": "existing_attempt_fails_closed"},
        )
        workspace = out / "submitted"
        await self.environment.download_dir("/workspace", workspace)
        if not workspace.is_dir() or any(p.is_symlink() for p in workspace.rglob("*")):
            raise ValueError("Submitted workspace is absent or contains symlinks")
        manifest = self.manifest["tasks"][task_id]
        if hashlib.sha256((workspace / "INSTRUCTIONS.md").read_bytes()).hexdigest() != manifest["prompt_sha256"]:
            raise ValueError("Submitted task instructions differ from frozen task")
        refs = self.source / "tasks" / task_id / "target_study"
        for name, digest in manifest["references"].items():
            # Existing manifest keys are target_study/<relative file>.
            rel = Path(name)
            if rel.is_absolute() or ".." in rel.parts or rel.parts[0] != "target_study":
                raise ValueError("Invalid reference manifest path")
            file = refs.joinpath(*rel.parts[1:])
            if file.is_symlink() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
                raise ValueError("Pinned creator reference mismatch")
        controller = Path(__file__).resolve().parents[4] / "benchmarks/researchclaw/guarded_score.py"
        result_path = out / "scientific-score.json"
        process = None
        start = None
        with (out / "controller.log").open("xb") as log:
            try:
                start = asyncio.create_task(
                    asyncio.create_subprocess_exec(
                        self.controller_python,
                        str(controller),
                        str(self.source),
                        str(workspace),
                        self.endpoint,
                        lineage,
                        task_id,
                        str(result_path),
                        stdout=log,
                        stderr=asyncio.subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
                process = await asyncio.shield(start)
                code = await asyncio.wait_for(process.wait(), timeout=self.task.config.verifier.timeout_sec)
                if code != 0:
                    raise RuntimeError(f"Creator scientific verifier failed with exit {code}; no reward")
            finally:
                if start is not None and process is None:
                    try:
                        process = await _settle(start)
                    except asyncio.CancelledError:
                        if start.done() and not start.cancelled() and start.exception() is None:
                            process = start.result()
                if process is not None and process.returncode is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(asyncio.shield(process.wait()), timeout=5)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        await asyncio.shield(process.wait())
        result = json.loads(result_path.read_text())
        if result["failures"] or result["reward"] is None:
            raise RuntimeError("Native judge failures cannot become a zero benchmark reward")
        return VerifierResult(rewards={"researchclaw_native_checklist": result["reward"]})
