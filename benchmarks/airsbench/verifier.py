# SPDX-License-Identifier: Apache-2.0
"""Score in Harbor's existing isolated verifier; never launch a third sandbox."""

import hashlib
import json
import math
from pathlib import Path

from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from benchmarks.airsbench.regrade import scorer_identity, validate_inputs


class ModelMissingSubmissionError(ValueError):
    """Complete solver output lacks the required scientific submission."""


class AIRSVerifier(BaseVerifier):
    def __init__(self, *, source_roots_json: str, **kwargs):
        super().__init__(**kwargs)
        self.roots = json.loads(source_roots_json)

    async def verify(self):
        if self.task.config.verifier.environment_mode != "separate":
            raise ValueError("AIRS requires separate verification")
        name = self.task.paths.task_dir.name
        source = Path(self.roots[name])
        output = self.trial_paths.verifier_dir / "native-score"
        output.mkdir(parents=True, exist_ok=False)
        workspace = output / "saved-workspace"
        artifact_manifest = json.loads((self.trial_paths.artifacts_dir / "manifest.json").read_text())
        captured = [entry for entry in artifact_manifest if entry["source"] == "/workspace"]
        if len(captured) != 1 or captured[0]["status"] not in ("ok", "empty"):
            raise RuntimeError("Incomplete native workspace artifact capture; not a model failure")
        # Harbor has restored the declared artifact into this separate environment.
        # A failed download is an infrastructure failure, never missing model output.
        await self.environment.download_dir_with_exclusions(
            source_dir="/workspace", target_dir=workspace, exclude=["data", "./data"]
        )
        task = validate_inputs(name, source, workspace)
        prediction = workspace / "submission.csv"
        if not prediction.is_file():
            (output / "failure.json").write_text(
                json.dumps(
                    {
                        "kind": "model_missing_submission",
                        "capture_complete": True,
                        "raw_mae": None,
                        "new_model_calls": 0,
                    }
                )
            )
            raise ModelMissingSubmissionError("Model omitted submission.csv; no creator MAE exists")
        before = hashlib.sha256(prediction.read_bytes()).hexdigest()
        identity = scorer_identity(task) | {
            "task_id": name,
            "image": task["image"],
            "prediction_sha256": before,
            "new_model_calls": 0,
            "metric_lower_is_better": True,
            "normalization": None,
            "native_verifier_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "execution_boundary": "Harbor separate verifier environment",
        }
        (output / "identity.json").write_text(json.dumps(identity, indent=2))
        # Only validated reference assets enter the verifier, after solver completion.
        await self.environment.upload_dir(source / "tests", "/tests")
        execution = await self.environment.exec("python /tests/check.py", timeout_sec=300)
        (output / "execution.json").write_text(
            json.dumps(
                {
                    "return_code": execution.return_code,
                    "stdout": execution.stdout,
                    "stderr": execution.stderr,
                },
                indent=2,
            )
        )
        if execution.return_code != 0:
            raise RuntimeError("Creator scorer rejected or failed; inspect diagnostics, not a scientific zero")
        metrics_path = output / "native-metrics.json"
        await self.environment.download_file("/logs/verifier/native-metrics.json", metrics_path)
        metrics = json.loads(metrics_path.read_text())
        mae = metrics["MAE"]
        if isinstance(mae, bool) or not isinstance(mae, (float, int)) or not math.isfinite(mae) or mae < 0:
            raise ValueError("Creator MAE must be finite and nonnegative")
        if metrics["rows"] != task["test_predictions"]:
            raise ValueError("Creator score did not cover the full test predictions")
        if hashlib.sha256(prediction.read_bytes()).hexdigest() != before:
            raise ValueError("Saved predictions changed during verification")
        (output / "score.json").write_text(json.dumps(identity | {"raw_mae": mae, "rows": metrics["rows"]}, indent=2))
        return VerifierResult(rewards={"reward": mae, "raw_mae": mae})
