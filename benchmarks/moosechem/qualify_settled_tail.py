# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline creator-prefix replay ending before an explicitly settled HTTP503 request."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from benchmarks.moosechem.native import MODEL, run
from benchmarks.moosechem.qualify_replay import ReplayRefused, StrictSavedClient


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TailReached(BaseException):
    """Successful offline boundary, never caught by creator Exception retry loops."""


class SettledSequence:
    def __init__(self, saved, evidence, gateway, repo):
        proof = json.loads(Path(evidence).read_text())
        if proof["saved_sha256"] != sha(saved) or proof["gateway_sha256"] != sha(gateway):
            raise ReplayRefused("Evidence file binding mismatch")
        expected = json.loads(Path(__file__).with_name("source-sha256.json").read_text())
        if proof["creator_sha256"] != expected:
            raise ReplayRefused("Creator source provenance mismatch")
        self.verify_creator(Path(repo), expected)
        self.rows = [json.loads(line) for line in Path(saved).read_text().splitlines()]
        if len(self.rows) < 2:
            raise ReplayRefused("Missing successful prefix")
        self.tail = self.rows[-1]
        if "response" in self.tail or self.tail.get("error_type") != "InternalServerError":
            raise ReplayRefused("Not a captured HTTP503 error tail")
        ledger = [json.loads(line) for line in Path(gateway).read_text().splitlines()]
        seen = set()
        for i, row in enumerate(self.rows):
            request = row["request"]
            if set(request) != {"model", "messages", "max_tokens", "temperature"} or request["model"] != MODEL:
                raise ReplayRefused("Unsupported request identity")
            wire = json.dumps(
                {k: request[k] for k in ("model", "messages", "max_tokens", "temperature")},
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            digest = hashlib.sha256(wire).hexdigest()
            starts = [r for r in ledger if r.get("event") == "started" and r.get("request_sha256") == digest]
            if len(starts) != 1 or starts[0]["request_id"] in seen:
                raise ReplayRefused("Ambiguous or missing gateway request")
            start = starts[0]
            seen.add(start["request_id"])
            ends = [r for r in ledger if r.get("request_id") == start["request_id"] and r.get("event") != "started"]
            if len(ends) != 1 or ends[0]["event"] != "finished":
                raise ReplayRefused("Gateway transport not settled")
            end = ends[0]
            if not row["started"] <= start["time"] <= end["time"] <= row["started"] + row["elapsed_seconds"]:
                raise ReplayRefused("Gateway chronology mismatch")
            if i == len(self.rows) - 1:
                if end["http_status"] != 503 or start["request_id"] != proof["tail_gateway_request_id"]:
                    raise ReplayRefused("Tail rejection mismatch")
            else:
                response = row.get("response", {})
                choices = response.get("choices", [])
                if (
                    end["http_status"] != 200
                    or end.get("provider_response_id") != response.get("id")
                    or response.get("model") != MODEL
                    or len(choices) != 1
                    or choices[0].get("finish_reason") != "stop"
                    or not choices[0].get("message", {}).get("content")
                    or "error_type" in row
                ):
                    raise ReplayRefused("Corrupt or unsuccessful saved response")
        self.index = 0

    def verify_creator(self, repo, expected):
        for name, digest in expected.items():
            if sha(repo / name) != digest:
                raise ReplayRefused("Creator source changed")


class PublicSettledSequence(SettledSequence):
    """Explicit public-only path, bound to the trusted export's exact manifest/task.

    The original full-repository constructor remains strict. The caller binds the
    public manifest hash in the execution identity after trusted full-source validation.
    """

    def __init__(self, saved, evidence, gateway, repo, *, manifest_sha256, task_id):
        self.public_identity = {"manifest_sha256": manifest_sha256, "task_id": task_id}
        super().__init__(saved, evidence, gateway, repo)
        if any(row["stage"] not in ("inspiration_screening", "hypothesis_generation") for row in self.rows):
            raise ReplayRefused("Reference-bearing verifier traces cannot enter public solver")

    def verify_creator(self, repo, expected):
        from benchmarks.moosechem.isolated import validate_public

        if sha(repo / "public-manifest.json") != self.public_identity["manifest_sha256"]:
            raise ReplayRefused("Public export binding mismatch")
        task = validate_public(
            repo, repo / "task.json", expected_manifest_sha256=self.public_identity["manifest_sha256"]
        )
        if task["original_task_id"] != self.public_identity["task_id"]:
            raise ReplayRefused("Public task identity mismatch")


class SettledClient(StrictSavedClient):
    def create(self, **request):
        if self.replay.index == len(self.replay.rows) - 1:
            effective = dict(request)
            if effective.get("max_tokens") != 8192:
                raise ReplayRefused("Creator budget changed")
            effective["max_tokens"] = self.budget
            if self.stage != self.replay.tail["stage"] or effective != self.replay.tail["request"]:
                raise ReplayRefused("Settled tail request divergence")
            raise TailReached()
        return super().create(**request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "saved", "evidence", "gateway", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--budget", type=int, choices=(8192, 16384), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    attempts = []

    def deny_network(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            attempts.append(event)
            raise ReplayRefused("Network forbidden")

    sys.addaudithook(deny_network)
    sequence = SettledSequence(args.saved, args.evidence, args.gateway, args.repo)
    try:
        run(
            args.repo,
            args.output,
            args.task_id,
            replay_path=args.saved,
            output_token_budget=args.budget,
            client_factory=SettledClient,
            replay_loader=lambda *a, **k: sequence,
        )
        raise ReplayRefused("Creator returned without reaching the rejected request")
    except TailReached:
        record = {
            "status": "exact_settled_tail_reached_no_dispatch",
            "replayed_calls": sequence.index,
            "new_model_calls": 0,
            "network_attempts": attempts,
            "network_boundary": "Python socket audit hook; not OS-wide isolation",
            "maintained_import_sha256": {
                str(Path(sys.modules[name].__file__).resolve()): sha(sys.modules[name].__file__)
                for name in ("benchmarks.moosechem.native", "benchmarks.moosechem.qualify_replay")
            },
            "controller_sha256": sha(__file__),
            "saved_sha256": sha(args.saved),
            "evidence_sha256": sha(args.evidence),
            "request": sequence.tail["request"],
            "stage": sequence.tail["stage"],
            "scientific_result": None,
        }
        (args.output / "qualification.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
