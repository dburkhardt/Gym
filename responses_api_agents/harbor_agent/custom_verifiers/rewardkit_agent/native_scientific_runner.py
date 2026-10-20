# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit judge-transport/failure-policy variant, native criteria and scores intact."""

import json
import os
import sys
import tomllib
from pathlib import Path

import rewardkit.judges as judges
from classify_native import classify
from rewardkit.agents import known_agents, register_agent
from rewardkit.runner import run
from scientific_backend import ScientificSuperJudge
from tool_loop import durable_write


register_agent(ScientificSuperJudge)

MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"
assert os.environ.get("GYM_RK_JUDGE_RUNTIME_APPROVED") == "1"
assert os.environ["REWARDKIT_JUDGE"] == "gym-super-scientific"
assert os.environ["REWARDKIT_MODEL"] == MODEL
# Native rubric/schema/weights/parsing unchanged; schema retry disabled explicitly.
judges._MAX_JUDGE_RETRIES = 1
native_configs = [tomllib.loads(p.read_text()) for p in Path("/tests").rglob("judge.toml")]
assert native_configs and all(c["judge"].get("timeout") == 600 for c in native_configs), (
    "Unreviewed native per-call timeout"
)
expected_calls = sum(
    len(c["criterion"]) if c["judge"].get("mode", "batched") == "individual" else 1 for c in native_configs
)
os.environ["GYM_RK_NATIVE_JUDGE_TIMEOUT"] = "600"
out = Path(sys.argv[1])
out.mkdir(exist_ok=False)
(out / "protocol.json").write_text(
    json.dumps(
        {
            "native_rewardkit": "0.1.7",
            "judge_model": MODEL,
            "judge_backend": "gym-super-scientific",
            "schema_attempts": 1,
            "native_timeout_preserved": True,
            "failure_policy_variant": True,
        }
    )
)
if ScientificSuperJudge.name not in known_agents():
    raise RuntimeError("Native scientific AgentCLI backend is not registered")
# RewardKit 0.1.7 discovery is intentionally one-shot: run() owns discovery.
try:
    result = run(
        "/tests",
        workspace="/app",
        output=out / "native-reward.json",
        max_concurrent_programmatic=1,
        max_concurrent_llm=1,
        max_concurrent_agent=1,
    )
except BaseException as e:
    (out / "failure.json").write_text(json.dumps({"mask_sample": True, "type": type(e).__name__, "reason": str(e)}))
    raise
assert set(result) == {"correctness", "process"}, "Native dimensions missing or unexpected"
details = json.loads((out / "reward-details.json").read_text())
classification = classify(result, details, os.environ["GYM_RK_JUDGE_RECEIPTS"], expected_calls)
durable_write(out / "classification.json", json.dumps(classification, indent=2).encode())
print(json.dumps(classification))
