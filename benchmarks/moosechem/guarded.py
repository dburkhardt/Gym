# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Creator subprocess using the shared task-scoped Gym model bridge."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai.types.chat import ChatCompletion

from benchmarks.moosechem.native import MODEL, TransportAbort, append_durable, run, write_failure


class GuardedTransport:
    def __init__(self, socket_path: str, token: str, lineage: str, digest: str) -> None:
        self.socket_path, self.token, self.lineage, self.digest = socket_path, token, lineage, digest
        self.ordinal = 0
        self.failed = False

    def factory(self, *, stage, trace, replay=None, output_token_budget=8192, **kwargs):
        if replay is not None:
            raise ValueError("Historical recovery is not implicit in a fresh native run")
        return SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **payload: self.create(stage, Path(trace), output_token_budget, payload)
                )
            )
        )

    def create(self, stage: str, trace: Path, budget: int, payload: dict) -> ChatCompletion:
        if self.failed:
            raise TransportAbort(RuntimeError("Prior dispatch requires reconciliation"))
        if payload.get("model") != MODEL or payload.get("max_tokens") != 8192 or budget not in (8192, 16384):
            raise TransportAbort(ValueError("Native model or output-budget contract mismatch"))
        payload = {**payload, "max_tokens": budget}
        ordinal = self.ordinal
        self.ordinal += 1
        self.failed = True
        record = {"stage": stage, "request": payload, "ordinal": ordinal}
        append_durable(trace.with_name("dispatch-intents.jsonl"), record)
        headers = {
            "Authorization": "Bearer " + self.token,
            "x-nemo-gym-logical-call-id": f"moosechem:{self.lineage}:{ordinal}",
            "x-nemo-gym-call-config-sha256": self.digest,
        }
        try:
            with httpx.Client(
                transport=httpx.HTTPTransport(uds=self.socket_path, retries=0), timeout=600, follow_redirects=False
            ) as client:
                response = client.post("http://gym-internal/chat/completions", json=payload, headers=headers)
                response.raise_for_status()
                raw = response.json()
            record["response"] = raw
            append_durable(trace, record)  # Preserve provider body before creator parsing.
            result = ChatCompletion.model_validate(raw)
            if result.model != MODEL:
                raise ValueError("Unexpected returned model")
            self.failed = False
            return result
        except Exception as exc:
            append_durable(
                trace.with_name("transport-failures.jsonl"),
                {
                    "ordinal": ordinal,
                    "error_type": type(exc).__name__,
                    "response_captured": "response" in record,
                },
            )
            raise TransportAbort(exc) from exc


def record_runtime_identity(output: Path) -> None:
    """Fail before dispatch if child imports or interpreter differ from selected files."""
    from benchmarks.moosechem import native

    expected = json.loads(Path(os.environ["MOOSE_EXECUTION_IDENTITY"]).read_text().splitlines()[0])
    actual = {
        "source": {
            str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (native.__file__, __file__)
        },
        "interpreter": {
            "path": str(Path(sys.executable).resolve()),
            "sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        },
        "python_version": sys.version,
    }
    append_durable(output / "runtime-imports.jsonl", actual)
    if actual["interpreter"] != expected["interpreter"] or any(
        expected["source"].get(path) != digest for path, digest in actual["source"].items()
    ):
        raise RuntimeError("Selected runtime/import identity mismatch before dispatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-token-budget", type=int, choices=[8192, 16384], default=8192)
    args = parser.parse_args()
    transport = GuardedTransport(
        os.environ["MOOSE_SOCKET"],
        os.environ["MOOSE_SOCKET_TOKEN"],
        os.environ["MOOSE_LINEAGE"],
        os.environ["MOOSE_CONFIG_SHA256"],
    )
    try:
        record_runtime_identity(args.output)
        result = run(
            args.repo,
            args.output,
            args.task_id,
            output_token_budget=args.output_token_budget,
            client_factory=transport.factory,
        )
        append_durable(args.output / "terminal.jsonl", {"result": result})
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
    except BaseException as exc:
        write_failure(args.output, exc)
        raise


if __name__ == "__main__":
    main()
