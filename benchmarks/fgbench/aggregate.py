# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report creator metrics and optionally the native per-metric projection."""

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.fgbench.projection import project
from benchmarks.scientific_agent_campaign.deterministic import module, score, validate_repo
from nemo_gym.result_disposition import summarize_dispositions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-report", type=Path)
    args = parser.parse_args()
    validate_repo("fgbench", args.repo)
    rows = [json.loads(line) for line in args.rollouts.read_text().splitlines()]
    answers = [
        "".join(
            c["text"]
            for m in r["response"]["output"]
            if m["type"] == "message"
            for c in m["content"]
            if c["type"] == "output_text"
        )
        for r in rows
    ]
    scorer = module(args.repo / "result_ananlysis/analysis.py")
    metadata = [r["verifier_metadata"] for r in rows]
    if set(scorer.task_name(r) for r in metadata) != set(scorer.TASK_COUNTS):
        raise ValueError(
            "Pilot does not cover all six creator task strata; report partial coverage, not full aggregate."
        )
    expected = scorer.evaluate(metadata, answers, "standard")
    args.output.write_text(json.dumps(expected, indent=2))
    if args.native_report:
        projected = []
        for row, answer in zip(rows, answers):
            task = {"uuid": row["uuid"], "verifier_metadata": row["verifier_metadata"]}
            details = score("fgbench", args.repo, task, answer)
            if details["reward"] != row["reward"]:
                raise ValueError("Saved raw reward differs from pinned creator recomputation")
            params = row["responses_create_params"]
            protocol = {
                "model": row["response"]["model"],
                "temperature": params.get("temperature"),
                "max_output_tokens": params.get("max_output_tokens"),
            }
            projected.append(
                {
                    **row,
                    "metadata": {
                        **(row.get("metadata") or {}),
                        **project(args.repo, task, details, protocol=protocol),
                    },
                }
            )
        report = summarize_dispositions(projected, expected_count=len(rows))
        report["saved_source"] = {
            "path": str(args.rollouts.resolve()),
            "sha256": hashlib.sha256(args.rollouts.read_bytes()).hexdigest(),
        }
        report["creator_metrics"] = expected
        report["qualification"] = (
            "Saved creator-score replay/projection; not new inference or independent source-item correctness review."
        )
        args.native_report.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
