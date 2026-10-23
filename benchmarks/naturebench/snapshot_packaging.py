# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-bound opt-in patch; never edits the scientific scorer or timeouts."""

import hashlib
import math
import tomllib
from pathlib import Path

from benchmarks.naturebench.packaging import MOLECULAR_TASK


EVALUATION_SHA256 = "1746282f390dd54761a91b7b5836b962e9483dc7394149dafd69dd51ac247272"


def patch_evaluation(source: str, *, capture_timeout_seconds: float) -> str:
    """Add per-request snapshot evidence to the exact known sidecar controller."""
    if (
        isinstance(capture_timeout_seconds, bool)
        or not math.isfinite(capture_timeout_seconds)
        or capture_timeout_seconds <= 0
    ):
        raise ValueError("Capture allowance must be finite and positive")
    if hashlib.sha256(source.encode()).hexdigest() != EVALUATION_SHA256:
        raise ValueError("Unqualified NatureBench evaluation controller source")
    source = source.replace(
        "from .timer import EffectiveTimer",
        "from .timer import EffectiveTimer\nfrom .evaluation_inputs import capture_inputs, SnapshotCaptureError",
    )
    source = source.replace(
        "self._submissions: list[dict[str, Any]] = []",
        "self._submissions: list[dict[str, Any]] = []\n        self._snapshot_attempt = 0\n        if (self.state_dir / 'evaluation_inputs').exists():\n            raise ValueError('Refuse reuse of prior evaluation input state')",
    )
    source = source.replace(
        "        self.timer.begin_evaluation()\n        try:\n            raw_scores = self._evaluator(self.output_dir)",
        """        self.timer.begin_evaluation()
        deadline = time.monotonic() + CAPTURE_TIMEOUT_SECONDS
        with self._state_lock:
            self._snapshot_attempt += 1
            attempt = self._snapshot_attempt
        snapshot = self.state_dir / "evaluation_inputs" / str(attempt)
        input_evidence = {"manifest": f"evaluation_inputs/{attempt}/manifest.json"}
        try:
            capture_inputs(self.output_dir, snapshot, deadline=deadline)
            if time.monotonic() >= deadline:
                raise SnapshotCaptureError("Evaluation input capture deadline exhausted")
            raw_scores = self._evaluator(snapshot / "output")""",
    )
    source = source.replace("            attempt = len(self._submissions) + 1\n", "")
    source = source.replace(
        '                    "timestamp": time.time(),',
        '                    "timestamp": time.time(),\n                    "input_evidence": input_evidence,',
    )
    source = source.replace(
        '                    "error": str(error),',
        '                    "error": str(error),\n                    "failure_kind": "evaluation_input_snapshot_failed" if isinstance(error, SnapshotCaptureError) else "evaluator_error",',
    )
    source = source.replace("CAPTURE_TIMEOUT_SECONDS", repr(float(capture_timeout_seconds)))
    return source


def install_snapshot_variant(task: Path, *, capture_timeout_seconds: float) -> None:
    """Patch only a freshly materialized molecular task and retain its inputs in artifacts."""
    if task.name != MOLECULAR_TASK:
        raise ValueError("Snapshot variant is qualified only for the molecular task")
    module = task / "environment/sidecar/naturebench_eval_service/evaluation.py"
    revised = patch_evaluation(module.read_text(), capture_timeout_seconds=capture_timeout_seconds)
    dockerfile = task / "environment/sidecar/Dockerfile"
    if (
        "COPY naturebench_eval_service /opt/naturebench/naturebench_eval_service"
        not in dockerfile.read_text().splitlines()
    ):
        raise ValueError("Sidecar build does not install the qualified controller")
    config = task / "task.toml"
    text = config.read_text()
    hooks = tomllib.loads(text).get("verifier", {}).get("collect", [])
    drains = [
        h["timeout_sec"]
        for h in hooks
        if h.get("service") == "naturebench-eval" and h.get("command") == "/opt/naturebench/bin/drain-and-finalize"
    ]
    if len(drains) != 1 or capture_timeout_seconds + 3600 >= drains[0]:
        raise ValueError("Capture allowance must fit existing collection allowance with positive overhead")
    anchor = '  { source = "/state/submissions.jsonl", service = "naturebench-eval" },'
    if text.count(anchor) != 1 or "/state/evaluation_inputs" in text:
        raise ValueError("Unqualified NatureBench artifact export configuration")
    helper = module.with_name("evaluation_inputs.py")
    with helper.open("xb") as stream:
        stream.write(Path(__file__).with_name("evaluation_inputs.py").read_bytes())
    module.write_text(revised)
    config.write_text(
        text.replace(anchor, anchor + '\n  { source = "/state/evaluation_inputs", service = "naturebench-eval" },')
    )
