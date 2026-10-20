# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Call the pinned creator's test-time chemistry reward without substitutions."""


def grade(text: str, *, solution: str, **metadata: object) -> dict:
    import os

    if solution.split("!:!")[0] in {"rxn_forward", "sol_eval"} and not os.environ.get("ETHER0_REMOTES_API_BASE_URL"):
        raise RuntimeError("This task requires the creator chemical oracle service; not configured")
    # The creator wheel omits these native BloomFilter inputs. Missing files
    # cause molbloom to segfault rather than raising FileNotFoundError.
    import hashlib
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec("ether0")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Pinned ether0 creator package is not installed; install resource requirements before scoring"
        )
    package = Path(spec.origin).parent
    hashes = {
        "fingerprints.bloom": "f9267f986c512f26131eb0219170c12e07ded9c1574c75233da5e3d5f3f00278",
        "rings.bloom": "b1b81b5dc5e59d2068742aafb0c217067e951260f56527998e902d37ac177a25",
    }
    for name, digest in hashes.items():
        path = package / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Missing or mismatched pinned creator asset: {path}; run stage_assets.py")
    from ether0.rewards import accuracy_reward

    diagnostics = []
    score = accuracy_reward([text], [solution], reasoning=False, test=True, metadata=diagnostics)[0]
    detail = diagnostics[0]
    if detail.get("reward_reason") == "reward_function_exception":
        raise RuntimeError("Upstream chemistry oracle failed; do not count as model error")
    return {"reward": float(score), "diagnostics": detail}
