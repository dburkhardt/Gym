# SPDX-License-Identifier: Apache-2.0
"""Run the creator outcome scorer in an owned, network-isolated verifier container.

This is outcome-only regrading, not completion of the process checklist.
"""

import hashlib
import json
import re
import uuid
from pathlib import Path

from benchmarks.biabench.regrade_outcome import prediction_hashes
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.lifecycle import _owned_sandbox


async def regrade_managed(*, source: Path, task: Path, predictions: Path, output: Path, image: str) -> dict:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Verifier image must be pinned by full Docker image digest")
    inputs = [path.resolve(strict=True) for path in (source, task, predictions)]
    if any(not path.is_dir() for path in inputs):
        raise ValueError("Creator source, task and saved predictions must be directories")
    destination = output.resolve()
    if any(
        destination == path or destination.is_relative_to(path) or path.is_relative_to(destination) for path in inputs
    ):
        raise ValueError("Verifier output must be separate from all input directories")
    before = prediction_hashes(inputs[2])
    destination.mkdir(parents=True, exist_ok=False)
    checkout = Path(__file__).resolve().parents[2]
    lineage = uuid.uuid4().hex
    identity = {
        "run": lineage,
        "image": image,
        "network": "none",
        "new_model_calls": 0,
        "scope": "outcome_only",
        "prediction_sha256": before,
        "controller_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("managed_regrade.py", "regrade_outcome.py", "prepare.py", "frozen_sources.json")
        },
    }
    (destination / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    sandbox = AsyncSandbox(
        {"docker": {"create": {"network": "none"}}},
        SandboxSpec(
            image=image,
            ttl_s=3720,
            resources=SandboxResources(cpu=2, memory_mib=4096),
            metadata={"workload": "gym-biabench-outcome-verifier", "run": lineage},
            env={"PYTHONPATH": "/gym", "PYTHONDONTWRITEBYTECODE": "1"},
            provider_options={
                "volumes": [
                    f"{checkout / 'benchmarks/biabench'}:/gym/benchmarks/biabench:ro",
                    f"{checkout / 'nemo_gym'}:/gym/nemo_gym:ro",
                    f"{inputs[0]}:/creator:ro",
                    f"{inputs[1]}:/task:ro",
                    f"{inputs[2]}:/predictions:ro",
                    f"{destination}:/results",
                ]
            },
        ),
    )
    async with _owned_sandbox(sandbox, destination / "lifecycle.json"):
        result = await sandbox.exec(
            "python /gym/benchmarks/biabench/regrade_outcome.py --source /creator "
            "--task /task --predictions /predictions --output /results/outcome.json",
            timeout_s=3600,
        )
        (destination / "stdout.log").write_text(result.stdout or "")
        (destination / "stderr.log").write_text(result.stderr or "")
        (destination / "execution.json").write_text(
            json.dumps({"return_code": result.return_code, "error_type": result.error_type}, indent=2) + "\n"
        )
        if result.return_code != 0:
            raise RuntimeError("BIABench verifier failed; retained artifacts are not a scientific zero")
    if prediction_hashes(inputs[2]) != before:
        raise ValueError("Saved predictions changed during managed verification")
    record = json.loads((destination / "outcome.json").read_text())
    if record["prediction_sha256"] != before or record["new_model_calls"] != 0:
        raise ValueError("Managed verifier output does not match saved prediction identity")
    return record
