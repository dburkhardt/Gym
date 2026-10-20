# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact-Super judge controller outside the isolated scientific shell sandbox.
No live execution authorized by this file. Separate readiness gate must approve endpoint.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import jsonschema


MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


def durable_write(path: Path, body: bytes) -> None:
    with path.open("xb") as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run_shell(command: str, timeout: float) -> dict:
    raise RuntimeError("Scientific shell must be an explicitly isolated executor")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--receipts")
    p.add_argument("--timeout", type=float, default=600)
    a = p.parse_args()
    assert a.timeout > 0
    assert a.model == MODEL
    # Explicit opt-in prevents accidental execution during offline import/control.
    assert os.environ.get("GYM_RK_JUDGE_RUNTIME_APPROVED") == "1", "Live runtime gate missing"
    endpoint = os.environ["GYM_RK_JUDGE_ENDPOINT"]
    assert "/ng-rollout/" in endpoint and endpoint.endswith("/v1"), "Guarded Gym capture route required"
    out = Path(a.receipts or os.environ["GYM_RK_JUDGE_RECEIPTS"])
    out.mkdir(parents=True, exist_ok=False)
    schema = json.loads(Path(a.schema).read_text())
    history = [{"role": "user", "content": a.prompt}]
    start = time.monotonic()
    for turn in range(40):
        remaining = a.timeout - (time.monotonic() - start)
        if remaining <= 0:
            raise TimeoutError("Native judge600s budget expired")
        request = {
            "model": MODEL,
            "input": history,
            "max_output_tokens": 16384,
            "tools": [
                {
                    "type": "function",
                    "name": "shell",
                    "description": "Inspect workspace files and run scientific checks.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                }
            ],
            "text": {"format": {"type": "json_schema", "name": "judge_response", "schema": schema, "strict": True}},
        }
        rp = out / f"{turn:03d}-request.json"
        durable_write(rp, json.dumps(request).encode())
        req = urllib.request.Request(
            endpoint.rstrip("/") + "/responses",
            data=json.dumps(request).encode(),
            headers={
                "Content-Type": "application/json",
                "x-nemo-gym-logical-call-id": os.environ["GYM_RK_TRIAL_LINEAGE"]
                + ":judge:"
                + out.name
                + ":"
                + str(turn),
                "x-nemo-gym-call-config-sha256": os.environ["GYM_RK_CALL_CONFIG_SHA256"],
            },
            method="POST",
        )
        # One dispatch, no retry. Preserve body before validation.
        try:
            with urllib.request.urlopen(req, timeout=remaining) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            durable_write(out / f"{turn:03d}-http-error-body.bin", e.read())
            durable_write(
                out / f"{turn:03d}-failure.json",
                json.dumps(
                    {
                        "type": type(e).__name__,
                        "http_status": e.code,
                        "dispatch_status": "http_error; provider completion not assumed",
                    }
                ).encode(),
            )
            raise
        except Exception as e:
            durable_write(
                out / f"{turn:03d}-failure.json",
                json.dumps({"type": type(e).__name__, "message": str(e), "dispatch_status": "unknown"}).encode(),
            )
            raise
        durable_write(out / f"{turn:03d}-response.json", body)
        res = json.loads(body)
        assert res.get("model") == MODEL, "Returned model mismatch"
        assert res.get("status") == "completed", "Incomplete judge response"
        usage = res.get("usage")
        assert isinstance(usage, dict), "Missing usage"
        assert all(
            type(usage.get(k)) is int and usage[k] >= 0 for k in ("input_tokens", "output_tokens", "total_tokens")
        ), "Invalid usage"
        assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"], "Usage mismatch"
        assert usage["output_tokens"] <= 16384, "Output budget exceeded"
        items = res.get("output", [])
        history.extend(items)
        calls = [x for x in items if x.get("type") == "function_call"]
        if not calls:
            texts = [
                c["text"]
                for x in items
                if x.get("type") == "message"
                for c in x.get("content", [])
                if c.get("type") == "output_text"
            ]
            if not texts:
                raise ValueError("Missing final judge output")
            # Native RewardKit validates criterion schema and normalizes final scores.
            final = "\n".join(texts)
            jsonschema.validate(json.loads(final), schema)
            print(final)
            return
        for i, c in enumerate(calls):
            assert c.get("name") == "shell"
            command = json.loads(c["arguments"])["command"]
            before = {"command": command, "cwd": str(Path.cwd()), "call_id": c["call_id"]}
            durable_write(out / f"{turn:03d}-tool-{i}-started.json", json.dumps(before).encode())
            remaining = a.timeout - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError("Native judge budget expired")
            result = run_shell(command, min(60, remaining))
            durable_write(out / f"{turn:03d}-tool-{i}.json", json.dumps(result).encode())
            history.append({"type": "function_call_output", "call_id": c["call_id"], "output": json.dumps(result)})
    raise RuntimeError("Judge40turn limit reached; mask result")


if __name__ == "__main__":
    main()
