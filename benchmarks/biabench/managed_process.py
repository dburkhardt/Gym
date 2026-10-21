# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned isolated process verifier; its only model access is Gym's guarded UDS bridge."""

import json
import re
import secrets
import tempfile
import uuid
from pathlib import Path

from benchmarks.biabench.prepare import validate
from benchmarks.biabench.process import MODEL, process_identity
from benchmarks.biabench.regrade_outcome import prediction_hashes
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.lifecycle import _owned_sandbox
from nemo_gym.sandbox.model_bridge import bridge


async def score_managed(
    *,
    source: Path,
    task: Path,
    submission: Path,
    output: Path,
    image: str,
    model_endpoint: str,
    model_name: str,
    timeout_s: int = 3600,
) -> dict:
    if model_name != MODEL or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Authorized model and immutable verifier image required")
    if not 0 < timeout_s <= 3600:
        raise ValueError("BIABench process verifier timeout must be within 3600 seconds")
    source, task, submission = (p.resolve(strict=True) for p in (source, task, submission))
    output = output.absolute()
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in (source, task, submission)):
        raise ValueError("Verifier output must be separate from immutable inputs")
    manifest = json.loads(Path(__file__).with_name("frozen_sources.json").read_text())
    validate(source, manifest["creator_files"])
    validate(task, manifest["task_files"])
    output.mkdir(parents=True, exist_ok=False)
    lineage = uuid.uuid4().hex
    identity, digest = process_identity(submission, lineage)
    (output / "identity.json").write_text(json.dumps(identity | {"digest": digest, "image": image}, indent=2))
    checkout = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="bia-bridge-") as temporary:
        socket_dir = Path(temporary)
        token = secrets.token_hex(32)
        request_path = output / "request.json"
        request_path.write_text(
            json.dumps(
                {
                    "source": "/creator",
                    "task": "/task",
                    "submission": "/submission",
                    "output": "/results/score",
                    "socket_path": "/bridge/model.sock",
                    "token": token,
                    "lineage": lineage,
                }
            )
        )
        request_path.chmod(0o600)
        sandbox = AsyncSandbox(
            {"docker": {"create": {"network": "none"}}},
            SandboxSpec(
                image=image,
                ttl_s=timeout_s + 120,
                resources=SandboxResources(cpu=2, memory_mib=4096),
                metadata={"workload": "gym-biabench-process", "run": lineage},
                env={"PYTHONPATH": "/gym", "PYTHONDONTWRITEBYTECODE": "1"},
                provider_options={
                    "volumes": [
                        f"{checkout / 'benchmarks/biabench'}:/gym/benchmarks/biabench:ro",
                        f"{checkout / 'nemo_gym'}:/gym/nemo_gym:ro",
                        f"{source}:/creator:ro",
                        f"{task}:/task:ro",
                        f"{submission}:/submission:ro",
                        f"{output}:/results",
                        f"{socket_dir}:/bridge:ro",
                    ]
                },
            ),
        )
        try:
            async with bridge(
                socket_dir / "model.sock",
                endpoint=model_endpoint,
                token=token,
                lineage=lineage,
                digest=digest,
                namespace="biabench_process",
                expected_model=MODEL,
            ):
                async with _owned_sandbox(sandbox, output / "lifecycle.json"):
                    result = await sandbox.exec(
                        "python -m benchmarks.biabench.process --request /results/request.json", timeout_s=timeout_s
                    )
                    (output / "execution.json").write_text(
                        json.dumps(
                            {
                                "return_code": result.return_code,
                                "error_type": result.error_type,
                                "stdout": result.stdout,
                                "stderr": result.stderr,
                            },
                            indent=2,
                        )
                    )
                    if result.return_code != 0:
                        raise RuntimeError(
                            "BIABench process verifier failed; retained partial results are not accepted"
                        )
        finally:
            request_path.unlink(missing_ok=True)
    if prediction_hashes(submission) != identity["submission_sha256"]:
        raise ValueError("Saved submission changed during process verification")
    child_identity = json.loads((output / "score/identity.json").read_text())
    if child_identity["digest"] != digest:
        raise ValueError("Parent and child scientific identities differ")
    return json.loads((output / "score/result.json").read_text())
