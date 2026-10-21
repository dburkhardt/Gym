# SPDX-License-Identifier: Apache-2.0
"""AIRS separate native verifier; raw MAE is lower-is-better, never normalized."""

import json
from pathlib import Path

from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from benchmarks.airsbench.regrade import regrade_managed


class AIRSVerifier(BaseVerifier):
    def __init__(self, *, source_roots_json: str, **kwargs):
        super().__init__(**kwargs)
        self.roots = json.loads(source_roots_json)

    async def verify(self):
        if self.task.config.verifier.environment_mode != "separate":
            raise ValueError("AIRS requires separate verification")
        name = self.task.paths.task_dir.name
        workspace = self.trial_paths.verifier_dir / "saved-workspace"
        await self.environment.download_dir("/workspace", workspace)
        result = await regrade_managed(
            task_id=name,
            source=Path(self.roots[name]),
            workspace=workspace,
            output=self.trial_paths.verifier_dir / "managed-score",
            capture_complete=True,
        )
        return VerifierResult(rewards={"reward": result["raw_mae"], "raw_mae": result["raw_mae"]})
