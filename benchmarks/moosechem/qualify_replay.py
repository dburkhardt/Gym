# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline creator-loop qualification; never falls through to provider inference."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from benchmarks.moosechem.native import append_durable, run


class ReplayRefused(BaseException):
    """Escape creator retry loops on incompatible or exhausted saved evidence."""


class OfflineSequence:
    def __init__(self, path, *, allow_transport_tail=False):
        if allow_transport_tail:
            raise ReplayRefused("Ambiguous transport replay is forbidden")
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        self.index = 0
        if not self.rows:
            raise ReplayRefused("Empty saved sequence")
        for i, row in enumerate(self.rows):
            if "response" not in row or any(key in row for key in ("error_type", "requires_reconciliation")):
                raise ReplayRefused(f"Unresolved saved call at {i}")
            choices = row["response"].get("choices", [])
            if len(choices) != 1 or choices[0].get("finish_reason") not in ("stop", "length"):
                raise ReplayRefused(f"Unsupported saved outcome at {i}")


class StrictSavedClient:
    def __init__(self, *, stage, trace, replay, output_token_budget, **kwargs):
        self.stage, self.trace, self.replay = stage, Path(trace), replay
        self.budget = output_token_budget
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        replay = self.replay
        if replay is None or replay.index >= len(replay.rows):
            raise ReplayRefused("Saved response sequence exhausted; inference forbidden")
        effective = dict(request)
        if effective.get("max_tokens") != 8192:
            raise ReplayRefused("Creator default token budget changed")
        effective["max_tokens"] = self.budget  # same declared historical run variant, no capture normalization
        row = replay.rows[replay.index]
        if row["stage"] != self.stage or row["request"] != effective:
            mismatch = {
                "index": replay.index,
                "expected_stage": row["stage"],
                "actual_stage": self.stage,
                "expected_request": row["request"],
                "actual_request": effective,
            }
            append_durable(self.trace.with_name("replay-mismatch.jsonl"), mismatch)
            raise ReplayRefused(f"Saved request divergence at index {replay.index}")
        if "response" not in row:
            raise ReplayRefused("Saved response absent or incomplete")
        from openai.types.chat import ChatCompletion

        value = ChatCompletion.model_validate(row["response"])
        append_durable(
            self.trace.with_name("replay-receipts.jsonl"),
            {"source_index": replay.index, "stage": self.stage, "inference": False},
        )
        replay.index += 1
        return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--saved", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--budget", type=int, choices=[8192, 16384], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    attempts = []

    def deny_network(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            attempts.append(event)
            raise ReplayRefused("Network operation forbidden in offline qualification")

    sys.addaudithook(deny_network)
    try:
        result = run(
            args.repo,
            args.output,
            args.task_id,
            replay_path=args.saved,
            output_token_budget=args.budget,
            client_factory=StrictSavedClient,
            replay_loader=OfflineSequence,
        )
        count = len(args.saved.read_text().splitlines())
        receipts = (args.output / "replay-receipts.jsonl").read_text().splitlines()
        if len(receipts) != count:
            raise ReplayRefused("Saved sequence not completely consumed")
        (args.output / "qualification.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "task_id": args.task_id,
                    "saved_sha256": hashlib.sha256(args.saved.read_bytes()).hexdigest(),
                    "replayed_calls": count,
                    "new_model_calls": 0,
                    "network_attempts": attempts,
                    "result": result,
                },
                indent=2,
            )
        )
    except BaseException as exc:
        (args.output / "qualification-failure.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "new_model_calls": 0,
                    "network_attempts": attempts,
                },
                indent=2,
            )
        )
        raise


if __name__ == "__main__":
    main()
