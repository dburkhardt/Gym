# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native SDK adapter to guarded Gym; one physical request per durable intent."""

import asyncio
import hashlib
import json
import time
from pathlib import Path

import httpx
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from nemo_gym.sandbox.guarded_chat_transport import publish
from nemo_gym.server_utils import request


class CapturedStream(httpx.AsyncByteStream):
    """Preserve each actual upstream chunk before returning it to the native parser."""

    def __init__(self, owner, response, stem: Path, *, streaming: bool):
        self.owner, self.response, self.stem, self.streaming = owner, response, stem, streaming
        self.closed = False

    async def __aiter__(self):
        chunks = []
        try:
            async with asyncio.timeout(600):
                async for chunk in self.response.content.iter_chunked(65536):
                    publish(
                        self.stem.with_name(self.stem.name + f".chunk{len(chunks):06d}.json"), {"hex": chunk.hex()}
                    )
                    chunks.append(chunk)
                    yield chunk
            body = b"".join(chunks)
            if self.streaming:
                events = [line[5:].strip() for line in body.splitlines() if line.startswith(b"data:")]
                if not events or events[-1] != b"[DONE]" or events.count(b"[DONE]") != 1:
                    raise RuntimeError("Native streaming response has no unique terminal DONE event")
                finished = False
                for event in events[:-1]:
                    parsed = ChatCompletionChunk.model_validate_json(event)
                    if parsed.model != self.owner.model:
                        raise ValueError("Native streaming model changed")
                    finished = finished or any(choice.finish_reason is not None for choice in parsed.choices)
                if not finished:
                    raise ValueError("Native stream ended without a completion finish reason")
            else:
                parsed = ChatCompletion.model_validate_json(body)
                if parsed.model != self.owner.model or not parsed.choices:
                    raise ValueError("Invalid native completion body")
            publish(
                self.stem.with_suffix(".complete.json"),
                {
                    "status": self.response.status,
                    "bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "completed_at": time.time(),
                },
            )
            self.owner.failed = False
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if not self.closed:
            self.closed = True
            self.response.release()
            self.owner.lock.release()


class NativeTransport(httpx.AsyncBaseTransport):
    """Use Gym's aiohttp transport underneath creator httpx/OpenAI interfaces."""

    def __init__(self, *, endpoint: str, lineage: str, digest: str, model: str, output: Path):
        if "/ng-rollout/" not in endpoint or not endpoint.endswith("/v1"):
            raise ValueError("Native capture-qualified endpoint required")
        output.mkdir(parents=True, exist_ok=False)
        self.endpoint, self.lineage, self.digest, self.model, self.output = endpoint, lineage, digest, model, output
        self.failed, self.ordinal = False, 0
        self.lock = asyncio.Lock()

    async def handle_async_request(self, incoming: httpx.Request) -> httpx.Response:
        await self.lock.acquire()
        response = None
        try:
            if self.failed:
                raise RuntimeError("Prior native dispatch unresolved; fresh retry forbidden")
            body = json.loads(await incoming.aread())
            if incoming.method != "POST" or not incoming.url.path.endswith("/chat/completions"):
                raise ValueError("Only native chat completions are allowed")
            if body.get("model") != self.model:
                raise ValueError("Native model identity changed")
            stem = self.output / f"{self.ordinal:06d}"
            headers = {
                "x-nemo-gym-logical-call-id": f"bioagent:{self.lineage}:{self.ordinal}",
                "x-nemo-gym-call-config-sha256": self.digest,
            }
            publish(stem.with_suffix(".request.json"), body)
            publish(
                stem.with_suffix(".intent.json"),
                {
                    "started_at": time.time(),
                    "headers": headers,
                    "request_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
                },
            )
            self.ordinal += 1
            self.failed = True
            response = await request(
                "POST",
                self.endpoint + "/chat/completions",
                json=body,
                headers=headers,
                allow_redirects=False,
                _max_num_tries=1,
                _max_connection_retries=0,
                _resend_after_disconnect=False,
            )
            if response.status != 200:
                raw = await response.read()
                publish(
                    stem.with_suffix(".error.json"), {"status": response.status, "body": raw.decode(errors="replace")}
                )
                raise RuntimeError(f"Native endpoint HTTP {response.status}; no automatic retry")
            stream = CapturedStream(self, response, stem, streaming=bool(body.get("stream")))
            return httpx.Response(response.status, headers=dict(response.headers), stream=stream, request=incoming)
        except BaseException:
            if response is not None:
                response.release()
            self.lock.release()
            raise

    async def aclose(self) -> None:
        # The shared aiohttp session belongs to Gym. Each response owns its release.
        return None
