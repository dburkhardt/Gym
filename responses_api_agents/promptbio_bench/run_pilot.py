# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired pilot CLI; paths and endpoint supplied at runtime, never baked into repository."""

import argparse
import json
from pathlib import Path

from responses_api_agents.promptbio_bench.runtime import download_task, grade, solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--scorer", type=Path)
    parser.add_argument("--scorer-python", default="python3")
    parser.add_argument("--judge-key-file", type=Path)
    parser.add_argument("--judge-model", default="gpt-5.4")
    parser.add_argument("--budget-ledger", type=Path)
    parser.add_argument("--budget-usd", type=float, default=100)
    args = parser.parse_args()
    for task_id in args.task:
        task = download_task(task_id, args.cache)
        runs = [
            solve(
                task,
                args.output,
                model=model,
                base_url=args.base_url,
                api_key=args.key_file.read_text().strip(),
                image=args.image,
            )
            for model in args.model
        ]
        scores = None
        if args.scorer:
            if not args.judge_key_file or not args.budget_ledger:
                parser.error("Grading requires --judge-key-file and --budget-ledger")
            scores = grade(
                task,
                runs,
                scorer_dir=args.scorer,
                python=args.scorer_python,
                output=args.output / task_id / "grading",
                judge_model=args.judge_model,
                judge_key_file=args.judge_key_file,
                budget_ledger=args.budget_ledger,
                budget_usd=args.budget_usd,
            )
        report = {
            "task_id": task_id,
            "models": args.model,
            "runs": [str(run) for run in runs],
            "scores": scores,
            "subset_result": True,
        }
        (args.output / task_id / "paired-result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))


if __name__ == "__main__":
    main()
