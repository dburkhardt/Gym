# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Container-only public solver or separate reference verifier entry point."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from benchmarks.moosechem.guarded import GuardedTransport, SettledContinuation
from benchmarks.moosechem.isolated import run_solver, run_verifier
from benchmarks.moosechem.native import append_durable, write_failure
from benchmarks.moosechem.qualify_settled_tail import PublicSettledSequence


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("solver", "verifier"))
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--budget", type=int, choices=(8192, 16384), required=True)
    p.add_argument("--public-manifest-sha256")
    p.add_argument("--continuation", action="store_true")
    p.add_argument("--ordinal", type=int, default=0)
    args = p.parse_args()
    if args.ordinal < 0 or (args.phase == "solver" and args.ordinal != 0):
        p.error("Invalid phase call ordinal")
    repo, out = Path("/repo"), Path("/workspace/scientific")
    if out.exists():
        raise ValueError("Existing native phase output cannot be replayed")
    expected = json.loads(os.environ["MOOSE_MODULE_HASHES"])
    actual = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in expected}
    if actual != expected:
        raise ValueError("Runtime source import binding changed")
    record = {
        "modules": actual,
        "python": sys.version,
        "phase": args.phase,
        "interpreter_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
    }
    append_durable(Path("/workspace/runtime-imports.jsonl"), record)
    options = (
        "/bridge/model.sock",
        os.environ["MOOSE_SOCKET_TOKEN"],
        os.environ["MOOSE_LINEAGE"],
        os.environ["MOOSE_CONFIG_SHA256"],
    )
    if args.continuation:
        if args.phase != "solver" or not args.public_manifest_sha256:
            p.error("Only the public solver can replay an explicitly settled prefix")
        sequence = PublicSettledSequence(
            Path("/settlement/saved.jsonl"),
            Path("/settlement/settlement.json"),
            Path("/settlement/gateway.jsonl"),
            repo,
            manifest_sha256=args.public_manifest_sha256,
            task_id=args.task_id,
        )
        transport = SettledContinuation(
            *options,
            sequence=sequence,
            output=out,
            evidence=Path("/settlement/settlement.json"),
            base_digest=os.environ["MOOSE_BASE_CONFIG_SHA256"],
        )
    else:
        transport = GuardedTransport(*options)
        transport.ordinal = args.ordinal
    try:
        if args.phase == "solver":
            result = run_solver(
                repo,
                out,
                repo / "task.json",
                client_factory=transport.factory,
                output_token_budget=args.budget,
                expected_manifest_sha256=args.public_manifest_sha256,
            )
            if args.continuation and not transport.tail_dispatched:
                raise RuntimeError("Creator returned before the exact settled boundary")
        else:
            result = run_verifier(
                repo,
                Path("/submission"),
                out,
                args.task_id,
                client_factory=transport.factory,
                output_token_budget=args.budget,
            )
        append_durable(Path("/workspace/terminal.jsonl"), {"result": result, "next_ordinal": transport.ordinal})
    except BaseException as exc:
        write_failure(Path("/workspace"), exc)
        raise


if __name__ == "__main__":
    main()
