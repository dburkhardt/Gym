# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from nemo_gym.sandbox.providers.docker import provider as docker


@pytest.mark.parametrize("running", [True, False])
async def test_reconcile_exact_owner_running_or_stopped(monkeypatch, running):
    monkeypatch.setattr(docker, "_require_docker", lambda: "/docker")
    provider = docker.DockerProvider()
    labels = {"nemo-gym.run": "a" * 32, "nemo-gym.workload": "desktop"}
    removed = []

    async def run(argv, **kwargs):
        if argv[1] == "ps":
            assert "label=nemo-gym.run=" + "a" * 32 in argv
            return 0, "immutable-id\n", ""
        if argv[1] == "rm":
            assert argv == ["/docker", "rm", "-f", "immutable-id"]
            removed.append(argv[-1])
            return 0, "", ""
        if removed:
            return 1, "", "No such container"
        return (
            0,
            json.dumps(
                {
                    "Id": "immutable-id",
                    "State": {"Running": running},
                    "Config": {"Labels": {**labels, docker.SANDBOX_LABEL: "1"}},
                }
            ),
            "",
        )

    monkeypatch.setattr(provider, "_run", run)
    receipt = await provider.reconcile_owned(labels)
    assert receipt["removed_ids"] == ["immutable-id"]


async def test_reconcile_wrong_owner_never_deletes(monkeypatch):
    monkeypatch.setattr(docker, "_require_docker", lambda: "/docker")
    provider = docker.DockerProvider()

    async def run(argv, **kwargs):
        assert argv[1] != "rm"
        if argv[1] == "ps":
            return 0, "id", ""
        return (
            0,
            json.dumps({"Id": "id", "Config": {"Labels": {docker.SANDBOX_LABEL: "1", "nemo-gym.run": "other"}}}),
            "",
        )

    monkeypatch.setattr(provider, "_run", run)
    with pytest.raises(RuntimeError, match="ownership mismatch"):
        await provider.reconcile_owned({"nemo-gym.run": "a" * 32, "nemo-gym.workload": "desktop"})
    with pytest.raises(ValueError, match="unique run"):
        await provider.reconcile_owned({"nemo-gym.workload": "desktop"})
