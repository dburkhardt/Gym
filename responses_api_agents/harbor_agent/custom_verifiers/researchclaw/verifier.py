# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harbor scientific verifier for the pinned ResearchClaw creator checklist."""

import hashlib
import json
import os
import shlex
import stat
import sys
import uuid
from pathlib import Path

from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from nemo_gym.sandbox.lifecycle import _publish
from nemo_gym.sandbox.providers.local.provider import LocalProvider
from nemo_gym.sandbox.scoring_workspace import scoring_workspace


def regular_tree_manifest(root: Path) -> dict[str, str]:
    """Reject links/devices/FIFOs before any host read of model-produced files."""
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("Submitted tree must be a regular directory")
    files = []
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise ValueError(f"Submitted tree contains non-regular entry: {path.relative_to(root)}")
            files.append(path)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


async def run_controller(command: list[str], *, timeout: float, out: Path, source_root: Path) -> None:
    """Use shared owned-process cleanup; this is trusted host scoring, not isolation."""
    provider = LocalProvider(concurrency=1)
    receipt = {"backend": "LocalProvider", "host_isolation": False, "completed": False}
    try:
        async with scoring_workspace(provider) as handle:
            receipt["workspace"] = str(handle.raw["workspace"])
            log_path = out / "controller.log"
            log_path.touch(exist_ok=False)
            result = await provider.exec(
                handle,
                shlex.join(command) + " > " + shlex.quote(str(log_path)) + " 2>&1",
                timeout_s=timeout,
                env={"PYTHONPATH": os.pathsep.join([str(source_root), *sys.path])},
                terminate_background_children=True,
            )
            with log_path.open("a") as log:
                log.write((result.stdout or "") + (result.stderr or ""))
            receipt.update(return_code=result.return_code, error_type=result.error_type)
            if result.error_type == "timeout":
                raise TimeoutError("Creator scientific verifier exceeded its native timeout; no reward")
            if result.return_code != 0:
                raise RuntimeError(f"Creator scientific verifier failed with exit {result.return_code}; no reward")
            receipt["completed"] = True
    except BaseException as error:
        receipt["error"] = type(error).__name__
        raise
    finally:
        receipt["owned_workspace_absent"] = not Path(receipt["workspace"]).exists() if "workspace" in receipt else None
        original = sys.exception()
        try:
            _publish(out / "controller-lifecycle.json", receipt)
        except Exception as error:
            if original is None:
                raise
            original.add_note(f"Controller lifecycle receipt failed: {error}")


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
        gym_provider_dispatch_guard: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if gym_model_name != "nvidia/nvidia/nemotron-3.5-super-vl":
            raise ValueError("ResearchClaw campaign requires exact Nemotron model")
        if "/ng-rollout/" not in gym_model_api_base or not gym_model_api_base.endswith("/v1"):
            raise ValueError("ResearchClaw requires guarded Gym capture endpoint")
        # Harbor injects this from the resolved server config before Ray dispatch.
        # A Ray worker must never initialize global config from its own CLI argv.
        if gym_provider_dispatch_guard is not True:
            raise ValueError("ResearchClaw requires provider_dispatch_guard enabled")
        self.controller_python = controller_python
        self.source = Path(source_root).resolve()
        self.manifest_path = Path(frozen_manifest).resolve()
        if hashlib.sha256(self.manifest_path.read_bytes()).hexdigest() != frozen_manifest_sha256:
            raise ValueError("Frozen task manifest hash mismatch")
        self.manifest_sha256 = frozen_manifest_sha256
        self.manifest = json.loads(self.manifest_path.read_text())
        self.endpoint = gym_model_api_base

    async def verify(self):
        if hashlib.sha256(self.manifest_path.read_bytes()).hexdigest() != self.manifest_sha256:
            raise ValueError("Frozen task manifest changed after initialization")
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
        workspace_manifest = regular_tree_manifest(workspace)
        manifest = self.manifest["tasks"][task_id]
        if hashlib.sha256((workspace / "INSTRUCTIONS.md").read_bytes()).hexdigest() != manifest["prompt_sha256"]:
            raise ValueError("Submitted task instructions differ from frozen task")
        refs = self.source / "tasks" / task_id / "target_study"
        reference_manifest = regular_tree_manifest(refs)
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
        _publish(
            out / "input-manifest.json",
            {
                "task_id": task_id,
                "lineage": lineage,
                "workspace": workspace_manifest,
                "references": reference_manifest,
                "creator_evaluation_files": regular_tree_manifest(self.source / "evaluation"),
                "frozen_manifest_sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
                "controller_sha256": hashlib.sha256(controller.read_bytes()).hexdigest(),
                "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
        )
        await run_controller(
            [
                self.controller_python,
                str(controller),
                str(self.source),
                str(workspace),
                self.endpoint,
                lineage,
                task_id,
                str(result_path),
            ],
            timeout=self.task.config.verifier.timeout_sec,
            out=out,
            source_root=controller.parents[2],
        )
        result = json.loads(result_path.read_text())
        if result["failures"] or result["reward"] is None:
            raise RuntimeError("Native judge failures cannot become a zero benchmark reward")
        return VerifierResult(rewards={"researchclaw_native_checklist": result["reward"]})
