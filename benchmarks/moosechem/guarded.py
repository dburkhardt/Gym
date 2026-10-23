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


def continuation_binding(sequence, evidence: Path, base_digest: str) -> tuple[dict, str]:
    """Bind settlement inputs and implementation into the existing bridge configuration."""
    from benchmarks.moosechem import native, qualify_replay, qualify_settled_tail

    if len(base_digest) != 64 or any(c not in "0123456789abcdef" for c in base_digest):
        raise ValueError("Base configuration SHA256 required")
    proof = json.loads(Path(evidence).read_text())
    identity = {
        "base_configuration_sha256": base_digest,
        "settlement_evidence_sha256": hashlib.sha256(Path(evidence).read_bytes()).hexdigest(),
        "saved_sha256": proof["saved_sha256"],
        "gateway_sha256": proof["gateway_sha256"],
        "creator_sha256": proof["creator_sha256"],
        "successful_prefix_count": len(sequence.rows) - 1,
        "output_token_budget": sequence.tail["request"]["max_tokens"],
        "rejected_request": sequence.tail["request"],
        "module_sha256": {
            Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (native.__file__, __file__, qualify_replay.__file__, qualify_settled_tail.__file__)
        },
    }
    return identity, hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class SettledContinuation(GuardedTransport):
    """Explicit, evidence-bound prefix replay; every new call uses the guarded bridge."""

    def __init__(self, *args, sequence, output: Path, evidence: Path, base_digest: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence = sequence
        self.tail_dispatched = False
        binding, expected_digest = continuation_binding(sequence, evidence, base_digest)
        if self.digest != expected_digest:
            raise ValueError("Continuation configuration digest mismatch")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if any(
            (self.output / name).exists()
            for name in ("model-traces.jsonl", "dispatch-intents.jsonl", "run-intent.json")
        ):
            raise ValueError("Existing attempt requires reconciliation")
        with (self.output / "continuation-claim.json").open("x") as claim:
            json.dump({"lineage": self.lineage}, claim)
        self.identity = binding | {"lineage": self.lineage, "configuration_sha256": self.digest}
        append_durable(self.output / "continuation-lineage.jsonl", self.identity)

    def factory(self, *, stage, trace, replay=None, output_token_budget=8192, **kwargs):
        if replay is not None:
            raise ValueError("A second replay source is forbidden")
        return SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **payload: self.continue_call(stage, Path(trace), output_token_budget, payload)
                )
            )
        )

    def continue_call(self, stage, trace, budget, payload):
        from benchmarks.moosechem.qualify_settled_tail import SettledClient, TailReached

        if budget != self.sequence.tail["request"]["max_tokens"]:
            raise TransportAbort(ValueError("Continuation budget changed"))
        if not self.tail_dispatched:
            client = SettledClient(stage=stage, trace=trace, replay=self.sequence, output_token_budget=budget)
            try:
                return client.create(**payload)
            except TailReached:
                # Mark before sending: failure may never fall back to replay or resend.
                self.tail_dispatched = True
                append_durable(
                    self.output / "continuation-boundary.jsonl",
                    {
                        "replayed_calls": self.sequence.index,
                        "next_guarded_ordinal": self.ordinal,
                        "lineage": self.lineage,
                    },
                )
        return super().create(stage, trace, budget, payload)


def record_runtime_identity(output: Path, *, continuation=False) -> None:
    """Fail before dispatch if child imports or interpreter differ from selected files."""
    from benchmarks.moosechem import native

    expected = json.loads(Path(os.environ["MOOSE_EXECUTION_IDENTITY"]).read_text().splitlines()[0])
    paths = (native.__file__, __file__)
    if continuation:
        from benchmarks.moosechem import qualify_replay, qualify_settled_tail

        paths += (qualify_replay.__file__, qualify_settled_tail.__file__)
    actual = {
        "source": {str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths},
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
    parser.add_argument("--settled-saved", type=Path)
    parser.add_argument("--settlement-evidence", type=Path)
    parser.add_argument("--settlement-gateway", type=Path)
    args = parser.parse_args()
    supplied = (args.settled_saved, args.settlement_evidence, args.settlement_gateway)
    if any(supplied) and not all(supplied):
        parser.error("All three explicit settlement paths are required")
    extra = {}
    transport_class = GuardedTransport
    if all(supplied):
        from benchmarks.moosechem.qualify_settled_tail import SettledSequence

        sequence = SettledSequence(args.settled_saved, args.settlement_evidence, args.settlement_gateway, args.repo)
        if args.output_token_budget != sequence.tail["request"]["max_tokens"]:
            parser.error("Continuation must preserve the historical output budget")
        transport_class = SettledContinuation
        extra = {
            "sequence": sequence,
            "output": args.output,
            "evidence": args.settlement_evidence,
            "base_digest": os.environ["MOOSE_BASE_CONFIG_SHA256"],
        }
    transport = transport_class(
        os.environ["MOOSE_SOCKET"],
        os.environ["MOOSE_SOCKET_TOKEN"],
        os.environ["MOOSE_LINEAGE"],
        os.environ["MOOSE_CONFIG_SHA256"],
        **extra,
    )
    try:
        record_runtime_identity(args.output, continuation=bool(extra))
        result = run(
            args.repo,
            args.output,
            args.task_id,
            output_token_budget=args.output_token_budget,
            client_factory=transport.factory,
        )
        if extra and not transport.tail_dispatched:
            raise RuntimeError("Creator returned before the verified continuation boundary")
        append_durable(args.output / "terminal.jsonl", {"result": result})
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
    except BaseException as exc:
        write_failure(args.output, exc)
        raise


if __name__ == "__main__":
    main()
