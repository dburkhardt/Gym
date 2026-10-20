# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run pinned creator stochastic scorer unchanged on saved parsed predictions.

Input predictions are JSON {task_id: [sentence indices]}; only locally produced
JSON is converted to the creator-required pickle. Dataset preserves creator order.
Canonical full-release comparisons require predictions for every293testrecord.
A subset replay remains explicitly a subset, with random.seed(42) at its start.
"""

import argparse
import json
import pickle
import subprocess
from pathlib import Path


PIN = "bf1d9633c694381c7b016fd56ee9f95f48593cc3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    upstream = Path(args.upstream).resolve()
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        raise ValueError(f"Expected creator revision {PIN}; got {revision}")
    predictions = json.loads(Path(args.predictions).read_text())
    if not all(isinstance(v, list) and all(type(i) is int for i in v) for v in predictions.values()):
        raise ValueError("Predictions must map IDs to integer lists")
    data = json.loads((upstream / "datasets/evidencebench_test_set.json").read_text())
    if not predictions or not set(predictions) <= set(data):
        raise ValueError("Empty or unknown prediction IDs")
    subset = {k: v for k, v in data.items() if k in predictions}
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "dataset.json").write_text(json.dumps(subset))
    (output / "predictions.pickle").write_bytes(pickle.dumps(predictions))
    import sys

    command = [
        sys.executable,
        str(upstream / "Evaluation/post_process/evaluate.py"),
        "--model_name",
        "nvidia/nvidia/nemotron-3.5-super-vl",
        "--input_path",
        str(output / "predictions.pickle"),
        "--dataset_path",
        str(output / "dataset.json"),
        "--exp_name",
        "creator-stochastic-ER10",
        "--logging_csv",
        str(output / "scores.csv"),
        "--limit_k",
        "10",
    ]
    run = subprocess.run(command, text=True, capture_output=True)
    (output / "stdout.txt").write_text(run.stdout)
    (output / "stderr.txt").write_text(run.stderr)
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "creator_pin": PIN,
                "count": len(subset),
                "full_release": len(subset) == len(data),
                "scoring": "unmodified creator script,100000 shuffles,global seed42",
                "returncode": run.returncode,
            },
            indent=2,
        )
    )
    run.check_returncode()


if __name__ == "__main__":
    main()
