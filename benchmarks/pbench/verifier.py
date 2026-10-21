# SPDX-License-Identifier: Apache-2.0
"""Native Harbor entry point for separate managed P-Bench scoring."""

from pathlib import Path

from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from benchmarks.pbench.managed_regrade import regrade_managed


class PBenchVerifier(BaseVerifier):
    def __init__(self, *, source: str, **kwargs):
        super().__init__(**kwargs)
        self.source = Path(source)

    async def verify(self):
        workspace = self.trial_paths.verifier_dir / "saved-workspace"
        await self.environment.download_dir("/workspace", workspace)
        result = await regrade_managed(
            source=self.source,
            task_id=self.task.paths.task_dir.name,
            workspace=workspace,
            output=self.trial_paths.verifier_dir / "managed-score",
            capture_complete=True,
        )
        return VerifierResult(rewards={"reward": result["strict"], "raw": result["raw"], "strict": result["strict"]})
