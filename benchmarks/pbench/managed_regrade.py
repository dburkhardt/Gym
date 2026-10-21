# SPDX-License-Identifier: Apache-2.0
"""Separate, network-disabled creator scorer over a complete saved workspace."""

import hashlib
import json
import stat
import uuid
from pathlib import Path

from benchmarks.pbench.native_prepare import IMAGE
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.lifecycle import _owned_sandbox


def validate_workspace(task_id: str, workspace: Path, *, capture_complete: bool = False) -> dict:
    manifest = json.loads(Path(__file__).with_name("frozen_sources.json").read_text())
    task = next((t for t in manifest["tasks"] if t["harbor_task"] == task_id), None)
    if task is None:
        raise ValueError("Unknown frozen P-Bench task")
    if workspace.is_symlink() or any(p.is_symlink() for p in workspace.rglob("*")):
        raise ValueError("Saved P-Bench workspace cannot contain symlinks")
    for entry in workspace.rglob("*"):
        mode = entry.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError("Saved P-Bench workspace contains a non-regular entry")
    if not workspace.is_dir() or not capture_complete:
        raise ValueError("Complete native workspace capture evidence is required")
    return task


async def regrade_managed(
    *, source: Path, task_id: str, workspace: Path, output: Path, capture_complete: bool = False
) -> dict:
    validate_workspace(task_id, workspace, capture_complete=capture_complete)
    source, workspace, output = source.resolve(strict=True), workspace.resolve(strict=True), output.resolve()
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in (source, workspace)):
        raise ValueError("Verifier output must be separate from source and saved workspace")
    output.mkdir(parents=True, exist_ok=False)
    controller = Path(__file__).parent.resolve()
    prediction = workspace / "result.json"
    before = hashlib.sha256(prediction.read_bytes()).hexdigest() if prediction.exists() else None
    lineage = uuid.uuid4().hex
    (output / "identity.json").write_text(
        json.dumps(
            {
                "run": lineage,
                "task_id": task_id,
                "image": IMAGE,
                "prediction_sha256": before,
                "new_model_calls": 0,
                "capture_complete": True,
                "controller_sha256": {
                    name: hashlib.sha256((controller / name).read_bytes()).hexdigest()
                    for name in ("score_worker.py", "regrade.py", "frozen_sources.json")
                },
            },
            indent=2,
        )
    )
    sandbox = AsyncSandbox(
        {"docker": {"create": {"network": "none"}}},
        SandboxSpec(
            image=IMAGE,
            ttl_s=120,
            resources=SandboxResources(cpu=1, memory_mib=512),
            metadata={"workload": "gym-pbench-verifier", "run": lineage},
            env={"PYTHONDONTWRITEBYTECODE": "1"},
            provider_options={
                "volumes": [
                    f"{controller}:/controller:ro",
                    f"{source}:/creator:ro",
                    f"{workspace}:/workspace:ro",
                    f"{output}:/results",
                ]
            },
        ),
    )
    (output / "request.json").write_text(json.dumps({"task_id": task_id}))
    async with _owned_sandbox(sandbox, output / "lifecycle.json"):
        execution = await sandbox.exec("python3 /controller/score_worker.py", timeout_s=60)
        (output / "execution.json").write_text(
            json.dumps(
                {"return_code": execution.return_code, "stdout": execution.stdout, "stderr": execution.stderr},
                indent=2,
            )
        )
        if execution.return_code != 0:
            raise RuntimeError("P-Bench verifier failed; preserved diagnostics are not a scientific zero")
    result = json.loads((output / "score.json").read_text())
    after = hashlib.sha256(prediction.read_bytes()).hexdigest() if prediction.exists() else None
    if before != after or result["prediction_sha256"] != before or result["new_model_calls"] != 0:
        raise ValueError("P-Bench verifier prediction identity mismatch")
    return result
