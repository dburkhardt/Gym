# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Call the pinned creator's test-time chemistry reward without substitutions."""


def grade(text: str, *, solution: str, **metadata: object) -> dict:
    import os

    if solution.split("!:!")[0] in {"rxn_forward", "sol_eval"} and not os.environ.get("ETHER0_REMOTES_API_BASE_URL"):
        raise RuntimeError("This task requires the creator chemical oracle service; not configured")
    from ether0.rewards import accuracy_reward

    diagnostics = []
    score = accuracy_reward([text], [solution], reasoning=False, test=True, metadata=diagnostics)[0]
    detail = diagnostics[0]
    if detail.get("reward_reason") == "reward_function_exception":
        raise RuntimeError("Upstream chemistry oracle failed; do not count as model error")
    return {"reward": float(score), "diagnostics": detail}
