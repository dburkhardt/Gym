# SPDX-License-Identifier: Apache-2.0
"""Single-attempt SDK transport to a task-scoped local model bridge.

Existing tool state is never silently reconstructed. A fresh transport requires
an unused artifact directory; uncertainty poisons the process until inspection.
"""

import asyncio
import json
import os
import time
from pathlib import Path

import httpx


def publish(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("x") as handle:
        json.dump(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class GuardedChatTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        *,
        socket_path: str,
        token: str,
        lineage: str,
        digest: str,
        namespace: str,
        model: str,
        output: Path,
        transport=None,
    ):
        if not namespace.isidentifier() or not lineage or not digest or not model:
            raise ValueError("Explicit call lineage, configuration and model required")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.inner = transport or httpx.AsyncHTTPTransport(uds=socket_path, retries=0)
        self.token, self.lineage, self.digest = token, lineage, digest
        self.namespace, self.model = namespace, model
        self.ordinal = 0
        self.failed = False
        self.lock = asyncio.Lock()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async with self.lock:
            if self.failed:
                raise RuntimeError("Unresolved native call; no automatic resend")
            payload = json.loads(await request.aread())
            if request.method != "POST" or not request.url.path.endswith("/chat/completions"):
                raise ValueError("Only chat completion requests are supported")
            if payload.get("model") != self.model or payload.get("stream"):
                raise ValueError("Unexpected model or streaming native request")
            stem = self.output / f"{self.ordinal:06d}"
            self.ordinal += 1
            self.failed = True
            publish(stem.with_suffix(".request.json"), payload)
            headers = {
                "Authorization": "Bearer " + self.token,
                "Content-Type": "application/json",
                "x-nemo-gym-logical-call-id": f"{self.namespace}:{self.lineage}:{self.ordinal - 1}",
                "x-nemo-gym-call-config-sha256": self.digest,
            }
            outgoing = httpx.Request(
                "POST", "http://gym-internal/chat/completions", headers=headers, content=json.dumps(payload).encode()
            )
            response = None
            started_at = time.time()
            try:
                response = await self.inner.handle_async_request(outgoing)
                content = await response.aread()
                publish(
                    stem.with_suffix(".response.json"),
                    {
                        "status": response.status_code,
                        "body": content.decode(),
                        "started_at": started_at,
                        "completed_at": time.time(),
                    },
                )
                if response.status_code != 200:
                    raise RuntimeError(f"Native model endpoint returned HTTP {response.status_code}; no SDK retry")
                body = json.loads(content)
                if body.get("model") != self.model or not body.get("choices"):
                    raise ValueError("Provider response has wrong model or no choices")
                # Preserve length/tool_calls finish reasons for the creator harness.
                # The complete HTTP response is durable even when generation truncated.
                self.failed = False
                return httpx.Response(response.status_code, headers=response.headers, content=content, request=request)
            finally:
                if response is not None:
                    await response.aclose()

    async def aclose(self):
        await self.inner.aclose()
