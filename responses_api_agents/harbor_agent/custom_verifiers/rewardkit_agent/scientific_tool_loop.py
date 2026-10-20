# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller outside the shell sandbox; no local-shell fallback is permitted."""

import os

import tool_loop
from docker_executor import execute


def configure() -> None:
    container = os.environ["GYM_RK_JUDGE_SANDBOX"]
    image = os.environ["GYM_RK_JUDGE_IMAGE"]
    assert image.startswith("sha256:") and len(image) == 71
    assert container and "/" not in container

    def shell(command, timeout):
        return execute(container, command, timeout, image, "gym-rewardkit-judge-scientific")

    tool_loop.run_shell = shell


if __name__ == "__main__":
    configure()
    tool_loop.main()
