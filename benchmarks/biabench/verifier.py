# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harbor verification hook retaining distinct creator outcome and process measures."""

import json
import shutil
from pathlib import Path

from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from benchmarks.biabench.managed_process import score_managed
from benchmarks.biabench.regrade_outcome import prediction_hashes


class BIAVerifier(BaseVerifier):
    def __init__(
        self,
        *,
        source: str,
        task_source: str,
        verifier_image: str,
        gym_model_api_base: str,
        gym_model_name: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.source, self.task_source = Path(source), Path(task_source)
        self.image, self.endpoint, self.model = verifier_image, gym_model_api_base, gym_model_name

    async def verify(self):
        if self.task.config.verifier.environment_mode != "separate":
            raise ValueError("BIABench requires artifact-only separate verification")
        manifest = json.loads((self.trial_paths.artifacts_dir / "manifest.json").read_text())
        captured = [r for r in manifest if r.get("source") == "/workspace"]
        if len(captured) != 1 or captured[0].get("status") not in ("ok", "empty"):
            raise RuntimeError("Incomplete solver artifact capture; no scientific score")
        output = self.trial_paths.verifier_dir / "biabench"
        output.mkdir(parents=True, exist_ok=False)
        workspace = output / "workspace"
        await self.environment.download_dir("/workspace", workspace)
        prediction_hashes(workspace)  # Reject links before constructing scorer evidence.
        submission = output / "submission"
        submission.mkdir()
        artifacts = workspace / "output"
        if artifacts.is_dir():
            shutil.copytree(artifacts, submission / "artifacts")
        else:
            (submission / "artifacts").mkdir()  # Creator outcome scorer decides missing-output behavior.
        logs = submission / "logs"
        logs.mkdir()
        for file in workspace.iterdir():
            if file.is_file():
                shutil.copyfile(file, logs / file.name)
        trajectory = self.trial_paths.agent_dir / "trajectory.json"
        if not trajectory.is_file():
            raise RuntimeError("Full native solver trajectory is required for process scoring")
        shutil.copyfile(trajectory, logs / "trajectory.json")
        instruction = (self.task.paths.task_dir / "instruction.md").read_text()
        (submission / "submission.json").write_text(
            json.dumps(
                {
                    "task_id": self.task.paths.task_dir.name,
                    "agent_name": "nemotron_harbor",
                    "task_instruction": instruction,
                }
            )
        )
        result = await score_managed(
            source=self.source,
            task=self.task_source,
            submission=submission,
            output=output / "managed-process",
            image=self.image,
            model_endpoint=self.endpoint,
            model_name=self.model,
        )
        scores = result["native_scores"]
        reward = scores.get("result_score")
        if reward is None:
            raise RuntimeError("Creator outcome unavailable; not a scientific zero")
        # Raw process null/unknown and every checklist item remain in retained native output.
        rewards = {"reward": reward, "outcome": reward}
        if scores.get("checklist_score") is not None:
            rewards["process"] = scores["checklist_score"]
        return VerifierResult(rewards=rewards)
