# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harbor0.24 verifier: native RewardKit controller and isolated scientific tool sandbox."""

import hashlib
import json
import uuid
from pathlib import Path

from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.lifecycle import _owned_sandbox as _owned_sandbox
from nemo_gym.sandbox.lifecycle import _publish as _publish
from nemo_gym.sandbox.lifecycle import _settle as _settle


class RewardKitAgentVerifier(BaseVerifier):
    def __init__(self, *, image_id: str, gym_model_api_base: str, gym_model_name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.image_id = image_id
        if self.task.config.environment.docker_image != image_id:
            raise ValueError("Solver and scientific verifier must use the same explicitly pinned task image")
        self.controller_dir = Path(__file__).parent.resolve()
        self.endpoint = gym_model_api_base
        if gym_model_name != "nvidia/nvidia/nemotron-3.5-super-vl":
            raise ValueError("RewardKit judge requires exact Nemotron Super model")
        if "/ng-rollout/" not in self.endpoint or not self.endpoint.endswith("/v1"):
            raise ValueError("RewardKit judge requires guarded Gym capture URL")
        if not self.image_id.startswith("sha256:") or len(self.image_id) != 71:
            raise ValueError("RewardKit scientific image must be digest pinned")

    async def verify(self) -> VerifierResult:
        from nemo_gym.provider_dispatch import configuration_digest

        cpu = self.task.config.environment.cpus
        memory = self.task.config.environment.memory_mb
        if cpu is None or memory is None:
            raise ValueError("Scientific CPU and memory limits must be explicit")
        tests = self.task.paths.tests_dir.resolve()
        if any(p.is_symlink() for p in tests.rglob("*")):
            raise ValueError("Verifier sources must not contain symlinks")
        digest = configuration_digest(
            {
                "task": self.task.config.model_dump(mode="json"),
                "image": self.image_id,
                "tests": {
                    str(p.relative_to(tests)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(tests.rglob("*"))
                    if p.is_file()
                },
                "controller": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.controller_dir.glob("*.py")
                },
            }
        )
        lineage = uuid.uuid4().hex
        out = self.trial_paths.verifier_dir / "native-controller"
        # An existing verifier attempt cannot be silently restarted against uncertain calls.
        # Reverification must use a new native Harbor trial/scorer record.
        out.mkdir(exist_ok=False, parents=True)
        _publish(
            out / "identity.json",
            {
                "lineage": lineage,
                "configuration_sha256": digest,
                "test_sources_sha256": {
                    str(p.relative_to(tests)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(tests.rglob("*"))
                    if p.is_file()
                },
                "recovery": "fail_closed_existing_verifier_directory",
                "trusted_controller_exposure": ["host_network", "docker_socket"],
                "tool_sandbox": "network_none_read_only_unprivileged",
            },
        )
        workspace = out / "submitted"
        await self.environment.download_dir("/app", workspace)
        # Preserve every submitted file; reject links rather than following arbitrary host paths.
        assert workspace.is_dir() and not any(p.is_symlink() for p in workspace.rglob("*"))
        # A duplicate read-only judge view retains bytes and executable intent, while allowing UID65534 to inspect all submitted files.
        workspace.chmod(0o755)
        for path in workspace.rglob("*"):
            path.chmod(0o755 if path.is_dir() or path.stat().st_mode & 0o111 else 0o644)
        trajectory = self.trial_paths.agent_dir / "trajectory.json"
        assert trajectory.is_file()
        controller_out = out / "controller"
        controller_out.mkdir()
        resources = SandboxResources(cpu=cpu, memory_mib=memory)
        ttl = int(self.task.config.verifier.timeout_sec) + 120
        scientific = AsyncSandbox(
            {
                "docker": {
                    "create": {
                        "network": "none",
                        "read_only": True,
                        "cap_drop": ["ALL"],
                        "security_opt": ["no-new-privileges"],
                    }
                }
            },
            SandboxSpec(
                image=self.image_id,
                ttl_s=ttl,
                resources=resources,
                metadata={"workload": "gym-rewardkit-judge-scientific", "run": lineage},
                provider_options={
                    "volumes": [
                        str(workspace.resolve()) + ":/app:ro",
                        str(trajectory.resolve()) + ":/logs/agent/trajectory.json:ro",
                    ],
                    "run_args": [
                        "--user",
                        "65534:65534",
                        "--tmpfs",
                        "/tmp:rw,nosuid,size=128m",
                        "--label",
                        "campaign.role=gym-rewardkit-judge-scientific",
                    ],
                },
            ),
        )
        async with _owned_sandbox(scientific, out / "scientific-lifecycle.json"):
            descriptor = await scientific.serialize()
            controller = AsyncSandbox(
                {"docker": {"create": {"network": "host"}}},
                SandboxSpec(
                    image=self.image_id,
                    ttl_s=ttl,
                    resources=resources,
                    metadata={"workload": "gym-rewardkit-controller", "run": lineage},
                    env={
                        "PYTHONPATH": "/controller",
                        "REWARDKIT_JUDGE": "gym-super-scientific",
                        "REWARDKIT_MODEL": "nvidia/nvidia/nemotron-3.5-super-vl",
                        "GYM_RK_JUDGE_RUNTIME_APPROVED": "1",
                        "GYM_RK_JUDGE_SANDBOX": descriptor["sandbox_id"],
                        "GYM_RK_JUDGE_IMAGE": self.image_id,
                        "GYM_RK_JUDGE_ENDPOINT": self.endpoint,
                        "GYM_RK_TRIAL_LINEAGE": lineage,
                        "GYM_RK_CALL_CONFIG_SHA256": digest,
                        "GYM_RK_JUDGE_RECEIPTS": "/control-output/judge-receipts",
                    },
                    provider_options={
                        "volumes": [
                            "/var/run/docker.sock:/var/run/docker.sock",
                            "/usr/bin/docker:/usr/bin/docker:ro",
                            str(workspace.resolve()) + ":/app:ro",
                            str(tests) + ":/tests:ro",
                            str(self.controller_dir) + ":/controller:ro",
                            str(controller_out.resolve()) + ":/control-output",
                        ]
                    },
                ),
            )
            async with _owned_sandbox(controller, out / "controller-lifecycle.json"):
                result = await controller.exec(
                    "uvx --offline --from harbor-rewardkit==0.1.7 python /controller/native_scientific_runner.py /control-output/result",
                    timeout_s=self.task.config.verifier.timeout_sec,
                )
                (out / "stdout.log").write_text(result.stdout)
                (out / "stderr.log").write_text(result.stderr)
                assert result.return_code == 0, "Native RewardKit controller failed"
                classification = json.loads((controller_out / "result/classification.json").read_text())
                assert not classification["mask_sample"], classification["failure_reasons"]
                return VerifierResult(rewards=classification["native_reward"])
