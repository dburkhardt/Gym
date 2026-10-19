# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local, serial OpenAI judge proxy with a durable reserve-before-send budget ledger.

Pricing: https://developers.openai.com/api/docs/models/gpt-5.4 (2026-10-19).
No cached-input discount is assumed. Long-context and image requests are rejected.
"""

import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


class Budget:
    def __init__(self, path: Path, limit: float):
        self.path, self.limit = path, limit
        self.entries = json.loads(path.read_text()) if path.exists() else []

    def save(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.entries, indent=2))
        temporary.replace(self.path)

    def reserve(self, payload: dict) -> int:
        if any(entry.get("status") == "reserved" for entry in self.entries):
            raise ValueError("Unreconciled paid request; inspect ledger before continuing")
        if payload.get("stream"):
            raise ValueError("Streaming not supported by budget meter")
        encoded = json.dumps(payload).encode()
        if b'"image_url"' in encoded or b'"input_image"' in encoded:
            raise ValueError("Image pricing not supported by this pilot meter")
        # Bytes upper-bound text tokens; doubled allowance includes structured-output scaffolding.
        input_bound = 2 * len(encoded) + 1000
        if input_bound > 272000:
            raise ValueError("Conservative input bound exceeds supported pricing tier")
        output_bound = min(int(payload.get("max_completion_tokens", payload.get("max_tokens", 16384))), 16384)
        payload.pop("max_tokens", None)
        payload["max_completion_tokens"] = output_bound
        reserve = input_bound * 2.50 / 1e6 + output_bound * 15 / 1e6
        if sum(entry["cost_usd"] for entry in self.entries) + reserve > self.limit:
            raise ValueError("Judge budget exhausted")
        self.entries.append(
            {"status": "reserved", "cost_usd": reserve, "input_bound": input_bound, "output_bound": output_bound}
        )
        self.save()
        return len(self.entries) - 1

    def settle(self, index: int, usage: dict):
        entry = self.entries[index]
        inputs, outputs = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if not isinstance(inputs, int) or not isinstance(outputs, int) or inputs < 0 or outputs < 0:
            raise ValueError("Missing or invalid usage; retain reservation and stop")
        if inputs > entry["input_bound"] or outputs > entry["output_bound"]:
            raise ValueError("Usage exceeded reserved bound; retain reservation and stop")
        entry.update(
            status="settled",
            input_tokens=inputs,
            output_tokens=outputs,
            cost_usd=inputs * 2.50 / 1e6 + outputs * 15 / 1e6,
        )
        self.save()


@contextmanager
def proxy(*, key_file: Path, ledger: Path, limit: float = 100, model: str = "gpt-5.4"):
    """Yield the loopback base URL; never give the upstream key to the evaluator."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    budget = Budget(ledger, limit)
    key = key_file.read_text().strip()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                if self.path != "/v1/chat/completions":
                    raise ValueError("Only chat/completions is metered")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 136000:
                    raise ValueError("Unsupported request size")
                payload = json.loads(self.rfile.read(size))
                if payload.get("model") != model:
                    raise ValueError("Unexpected judge model")
                index = budget.reserve(payload)
                request = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=600) as response:
                    result = json.load(response)
                budget.settle(index, result.get("usage", {}))
                body, status = json.dumps(result).encode(), 200
            except Exception as exc:
                # Uncertain upstream outcomes remain reserved; retries cannot spend through them.
                body = json.dumps({"error": {"message": str(exc), "type": "budget_proxy_error"}}).encode()
                status = 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
