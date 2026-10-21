# SPDX-License-Identifier: Apache-2.0
"""Run the pinned AIRS creator MAE scorer against preserved predictions only."""

import hashlib
import json
import math
import stat
import uuid
from pathlib import Path

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.lifecycle import _owned_sandbox


def validate_inputs(task_id: str, source: Path, workspace: Path) -> dict:
    manifest = json.loads(Path(__file__).with_name("frozen_sources.json").read_text())
    task = next((t for t in manifest["tasks"] if t["name"] == task_id), None)
    if task is None:
        raise ValueError("Unknown frozen AIRS research task")
    if workspace.is_symlink() or not workspace.is_dir():
        raise ValueError("AIRS saved workspace must be a real directory")
    for p in workspace.rglob("*"):
        if not (stat.S_ISREG(p.lstat().st_mode) or stat.S_ISDIR(p.lstat().st_mode)):
            raise ValueError("AIRS saved workspace contains a symlink or special file")
    for name, digest in task["files"].items():
        if not name.startswith("tests/"):
            continue
        p = source / name
        if p.is_symlink() or not p.resolve().is_relative_to(source.resolve()) or not p.is_file():
            raise ValueError(f"Missing or unsafe AIRS scorer asset: {name}")
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed AIRS scorer asset: {name}")
    return task


async def regrade_managed(
    *, task_id: str, source: Path, workspace: Path, output: Path, capture_complete: bool = False
) -> dict:
    if not capture_complete:
        raise ValueError("Complete saved-workspace capture evidence is required")
    task = validate_inputs(task_id, source, workspace)
    source, workspace, output = source.resolve(), workspace.resolve(), output.resolve()
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in (source, workspace)):
        raise ValueError("Verifier output must be separate from sources and predictions")
    output.mkdir(parents=True, exist_ok=False)
    prediction = workspace / "submission.csv"
    if not prediction.is_file():
        (output / "failure.json").write_text(
            json.dumps(
                {"kind": "model_missing_submission", "capture_complete": True, "raw_mae": None, "new_model_calls": 0}
            )
            + "\n"
        )
        raise ValueError("Model omitted submission.csv; no creator MAE exists")
    before = hashlib.sha256(prediction.read_bytes()).hexdigest()
    identity = {
        "task_id": task_id,
        "image": task["image"],
        "prediction_sha256": before,
        "new_model_calls": 0,
        "metric_lower_is_better": True,
        "normalization": None,
    }
    (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    sandbox = AsyncSandbox(
        {"docker": {"create": {"network": "none"}}},
        SandboxSpec(
            image=task["image"],
            ttl_s=360,
            resources=SandboxResources(cpu=1, memory_mib=2048),
            metadata={"workload": "gym-airs-verifier", "run": uuid.uuid4().hex},
            env={"PYTHONDONTWRITEBYTECODE": "1"},
            provider_options={
                "volumes": [f"{source / 'tests'}:/tests:ro", f"{workspace}:/workspace:ro", f"{output}:/logs/verifier"]
            },
        ),
    )
    async with _owned_sandbox(sandbox, output / "lifecycle.json"):
        execution = await sandbox.exec("python /tests/check.py", timeout_s=300)
        (output / "execution.json").write_text(
            json.dumps(
                {"return_code": execution.return_code, "stdout": execution.stdout, "stderr": execution.stderr},
                indent=2,
            )
            + "\n"
        )
        if execution.return_code != 0:
            raise RuntimeError("Creator scorer rejected or failed; inspect diagnostics, not a scientific zero")
    metrics = json.loads((output / "native-metrics.json").read_text())
    mae = metrics["MAE"]
    if isinstance(mae, bool) or not isinstance(mae, (float, int)) or not math.isfinite(mae) or mae < 0:
        raise ValueError("Creator MAE must be finite and nonnegative")
    if metrics["rows"] != task["test_predictions"]:
        raise ValueError("Creator score did not cover the full test predictions")
    if hashlib.sha256(prediction.read_bytes()).hexdigest() != before:
        raise ValueError("Saved predictions changed during verification")
    result = identity | {"raw_mae": mae, "rows": metrics["rows"]}
    (output / "score.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
