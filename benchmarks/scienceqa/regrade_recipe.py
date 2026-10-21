# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordered saved-output scorer; never sends model requests."""

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.scienceqa.creator_recipe import score_ordered
from nemo_gym.openai_utils import NeMoGymCompletion, NeMoGymResponse
from nemo_gym.provider_dispatch import configuration_digest


def regrade(repo: Path, packet_path: Path, results_path: Path) -> dict:
    packet_bytes, results_bytes = packet_path.read_bytes(), results_path.read_bytes()
    packet = json.loads(packet_bytes)
    rows = [json.loads(line) for line in results_bytes.splitlines() if line.strip()]
    ids = [str(row.get("task_id", row.get("uuid"))) for row in rows]
    expected = [row["task_id"] for row in packet["solver_requests"]]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Saved outputs must contain exactly the prepared task IDs")
    indexed = dict(zip(ids, rows))
    outputs = []
    for request in packet["solver_requests"]:
        row = indexed[request["task_id"]]
        serialized = row["responses_create_params"]["input"]
        if len(serialized) != 1 or json.loads(serialized[0]["content"]) != request["request"]:
            raise ValueError("Saved row literal request differs from creator recipe")
        response = NeMoGymResponse.model_validate(row["response"])
        raw = NeMoGymCompletion.model_validate(row["response"]["raw_completion"])
        actual_request = row["response"]["raw_completion_request"]
        if actual_request != request["request"] | {"model": raw.model}:
            raise ValueError("Captured completion request differs from creator recipe")
        if raw.model != response.model or raw.id != response.id or raw.choices[0].text != response.output_text:
            raise ValueError("Completion response identity/text mismatch")
        if raw.choices[0].finish_reason == "content_filter":
            raise ValueError("Content-filtered completion requires separate qualification")
        outputs.append(response.output_text)
    identity = {
        "packet_sha256": hashlib.sha256(packet_bytes).hexdigest(),
        "results_sha256": hashlib.sha256(results_bytes).hexdigest(),
        "scorer_sha256": hashlib.sha256(Path(__file__).with_name("creator_recipe.py").read_bytes()).hexdigest(),
        "order": expected,
    }
    return {
        "scorer_lineage": configuration_digest(identity),
        "identity": identity,
        "rng_scope": packet["rng_scope"],
        "results": score_ordered(repo, packet, outputs),
        "model_calls": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = regrade(args.repo, args.packet, args.results)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
