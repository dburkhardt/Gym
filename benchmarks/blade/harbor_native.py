# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native BLADE solver and separate reference-bearing scorer on shared sandbox lifecycle."""

import hashlib
import json
import math
import secrets
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from benchmarks.blade.scoped_bridge import bridge
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.provider_dispatch import configuration_digest
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from responses_api_agents.harbor_agent.custom_verifiers.rewardkit_agent.verifier import _owned_sandbox, _publish


MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"
IMAGE = "sha256:2d642ec0d5e668a7194e78665c58e2b42a7ef9b5b3dbb8ee3ecd0a3fbc4da829"
DATASETS = ("crofoot", "panda_nuts", "amtl")


def validate_staged(root: Path) -> None:
    manifest = json.loads(Path(__file__).with_name("frozen_sources.json").read_text())
    expected = {}
    for relative, sha in manifest["files"].items():
        parts = Path(relative).parts
        if relative == "blade_bench/datasets/fertility/info.json":
            role = "solver"
        elif relative.startswith("blade_bench/datasets/"):
            if len(parts) != 4 or parts[2] not in DATASETS:
                continue
            role = "solver" if parts[3] in ("data.csv", "info.json") else "references"
        elif (
            relative.startswith("blade_bench/")
            and relative.endswith(".py")
            or relative in ("pyproject.toml", "LICENSE")
        ):
            role = "solver"
        else:
            continue
        expected[f"{role}/{relative}"] = sha
    for name, sha in expected.items():
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError(f"Pinned BLADE staged source mismatch: {name}")
    for role in ("solver", "references"):
        for path in (root / role).rglob("*"):
            if path.is_symlink():
                raise ValueError("BLADE staged source contains symlink")
            if (
                path.is_file()
                and not path.name.endswith(".stage.lock")
                and str(path.relative_to(root)) not in expected
            ):
                raise ValueError(f"Unexpected BLADE staged file: {path}")


async def run_phase(
    *, root: Path, dataset: str, phase: str, endpoint: str, output: Path, submitted: Path | None = None
) -> Path:
    if dataset not in DATASETS or phase not in ("solver", "scorer"):
        raise ValueError("Task or native phase is not frozen")
    if (
        not get_global_config_dict().get("provider_dispatch_guard")
        or "/ng-rollout/" not in endpoint
        or not endpoint.endswith("/v1")
    ):
        raise ValueError("Native BLADE requires guarded Gym model service")
    validate_staged(root)
    output.mkdir(parents=True, exist_ok=False)
    lineage = uuid.uuid4().hex
    controller = Path(__file__).parent.resolve()
    digest = configuration_digest(
        {
            "image": IMAGE,
            "dataset": dataset,
            "phase": phase,
            "controller": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in controller.glob("*.py")},
        }
    )
    _publish(
        output / "identity.json",
        {
            "lineage": lineage,
            "config_sha256": digest,
            "phase": phase,
            "recovery": "existing_attempt_fails_closed",
            "network": "none; task-scoped UDS model bridge only",
        },
    )
    volumes = [
        f"{root.resolve()}/solver:/opt/blade:ro",
        f"{controller}/native_phase.py:/controller/native_phase.py:ro",
        f"{controller}/llm-config.yaml:/controller/llm-config.yaml:ro",
        f"{controller}/frozen_sources.json:/controller/frozen_sources.json:ro",
        f"{output.resolve()}:/output",
    ]
    if phase == "scorer":
        if submitted is None or not submitted.is_dir() or any(p.is_symlink() for p in submitted.rglob("*")):
            raise ValueError("Native submitted solver artifact directory missing or contains symlinks")
        volumes += [f"{submitted.resolve()}:/submitted:ro"]
        for name in ("annotations.csv", "mcq_dataset.json"):
            path = root / "references/blade_bench/datasets" / dataset / name
            if path.exists():
                volumes.append(f"{path.resolve()}:/opt/blade/blade_bench/datasets/{dataset}/{name}:ro")
    with TemporaryDirectory(prefix="blade-uds-", dir="/tmp") as socket_dir:
        token = secrets.token_urlsafe(32)
        async with bridge(
            Path(socket_dir) / "model.sock", endpoint=endpoint, token=token, lineage=lineage, digest=digest
        ):
            sandbox = AsyncSandbox(
                {"docker": {"create": {"network": "none"}}},
                SandboxSpec(
                    image=IMAGE,
                    ttl_s=3720,
                    resources=SandboxResources(
                        cpu=2 if phase == "solver" else 1, memory_mib=4096 if phase == "solver" else 2048
                    ),
                    metadata={"workload": "gym-blade-" + phase, "run": lineage},
                    env={
                        "PYTHONPATH": "/opt/blade",
                        "OPENAI_API_KEY": "gym-internal-placeholder",
                        "LLM_CONFIG_PATH": "/controller/llm-config.yaml",
                        "BLADE_ENDPOINT": "http://gym-internal/ng-rollout/task/v1",
                        "BLADE_SOCKET": "/model-socket/model.sock",
                        "BLADE_TOKEN": token,
                        "BLADE_LINEAGE": lineage,
                        "BLADE_CONFIG_SHA256": digest,
                    },
                    provider_options={"volumes": volumes + [f"{socket_dir}:/model-socket:ro"]},
                ),
            )
            async with _owned_sandbox(sandbox, output / "lifecycle.json"):
                result = await sandbox.exec(
                    f"python /controller/native_phase.py {phase} {dataset} /output", timeout_s=3600
                )
                (output / "stdout.log").write_text(result.stdout or "")
                (output / "stderr.log").write_text(result.stderr or "")
                if result.return_code != 0:
                    raise RuntimeError("Native BLADE phase failed; preserved artifacts are not a scientific zero")
    return output / phase


class BladeNativeAgent(BaseAgent):
    def __init__(self, *, prepared_root: str, api_base: str, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.model_name != MODEL:
            raise ValueError("Native BLADE requires exact authorized model")
        self.root, self.endpoint = Path(prepared_root), api_base

    @staticmethod
    def name() -> str:
        return "blade-native"

    def version(self) -> str:
        return "6118fa8d5007b91aa8c91c518182db82446a4547"

    async def setup(self, environment: BaseEnvironment) -> None:
        validate_staged(self.root)

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        dataset = instruction.strip()
        result = await run_phase(
            root=self.root,
            dataset=dataset,
            phase="solver",
            endpoint=self.endpoint,
            output=self.logs_dir / "native-solver",
        )
        if not (result / "multirun_analyses.json").is_file():
            raise RuntimeError("Native BLADE analysis output missing")
        await environment.upload_dir(result, "/workspace/solver")


class BladeNativeVerifier(BaseVerifier):
    def __init__(self, *, prepared_root: str, gym_model_api_base: str, gym_model_name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        if gym_model_name != MODEL:
            raise ValueError("BLADE matching roles require exact authorized model")
        self.root, self.endpoint = Path(prepared_root), gym_model_api_base

    async def verify(self) -> VerifierResult:
        dataset = self.task.paths.task_dir.name
        submitted = self.trial_paths.verifier_dir / "submitted"
        await self.environment.download_dir("/workspace/solver", submitted)
        result = await run_phase(
            root=self.root,
            dataset=dataset,
            phase="scorer",
            endpoint=self.endpoint,
            output=self.trial_paths.verifier_dir / "native-scorer",
            submitted=submitted,
        )
        metrics = json.loads((result / "eval_metrics.json").read_text())
        if metrics.get("status") != [5]:
            raise RuntimeError("Native BLADE scorer did not finish successfully; no scientific zero")
        rewards = {
            key: float(value)
            for key, value in metrics.items()
            if key.startswith(("hit_rate_", "average_coverage_"))
            and isinstance(value, (int, float))
            and math.isfinite(value)
        }
        if not rewards:
            raise RuntimeError("Native BLADE metric vector absent")
        return VerifierResult(rewards=rewards)
