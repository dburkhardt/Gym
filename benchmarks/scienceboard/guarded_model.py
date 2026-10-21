# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route the creator Model's exact OpenAI payload through the shared durable guard."""

import asyncio
import json
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator

import httpx
import requests

from benchmarks.scienceboard.prepare import MANIFEST, checked_source


PROTOCOL_VARIANT = "creator-openai-guarded-single-attempt-no-overflow-trimming-v1"
from nemo_gym.sandbox.guarded_chat_transport import GuardedSyncChatTransport


class CapturedModelActionError(ValueError):
    """A durable model answer failed to provide a textual action, not infrastructure."""


class BoundedUDS(httpx.AsyncBaseTransport):
    """Bound connection and full body while retaining shared guard ownership."""

    def __init__(self, socket_path: str, timeout: float = 480, inner: httpx.AsyncBaseTransport | None = None) -> None:
        self.timeout = timeout
        self.inner = inner or httpx.AsyncHTTPTransport(uds=socket_path, retries=0)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = None
        try:
            async with asyncio.timeout(self.timeout):
                response = await self.inner.handle_async_request(request)
                body = await response.aread()
                return httpx.Response(response.status_code, headers=response.headers, content=body, request=request)
        finally:
            if response is not None:
                await response.aclose()

    async def aclose(self) -> None:
        await self.inner.aclose()


class CreatorModelHTTP:
    """Only the creator model module receives this requests-compatible facade.

    Desktop/application requests are untouched. The creator supplies its complete
    payload, including screenshot bytes and sampling defaults; no prompt conversion
    or retry wrapper is introduced. The shared guard persists every physical call.
    """

    def __init__(
        self,
        *,
        socket_path: str,
        token: str,
        lineage: str,
        digest: str,
        model: str,
        output: Path,
        inner: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.bounded = BoundedUDS(socket_path, inner=inner)
        self.transport = GuardedSyncChatTransport(
            socket_path=socket_path,
            token=token,
            lineage=lineage,
            digest=digest,
            namespace="scienceboard",
            model=model,
            output=output,
            transport=self.bounded,
        )
        self.no_resend = False
        self.outcome = "not_dispatched"

    def post(self, url: str, *, headers: dict, proxies: dict | None, json: dict, timeout: float) -> requests.Response:
        if self.no_resend:
            raise RuntimeError("Prior creator model request failed; automatic retry forbidden")
        if url != "http://gym-internal/chat/completions" or proxies or not 0 < timeout <= 480:
            raise ValueError("Unexpected creator model route/proxy/timeout")
        if set(headers) - {"Content-Type"}:
            raise ValueError("Creator model must not receive upstream credentials")
        self.bounded.timeout = timeout
        self.no_resend = True
        self.outcome = "transport_unresolved"
        request = httpx.Request("POST", url, json=json)
        response = self.transport.handle_request(request)
        native = requests.Response()
        native.status_code = response.status_code
        native._content = response.content
        native.headers.update(response.headers)
        body = native.json()
        # Creator access() catches parse errors and retries; do not let a null model
        # answer silently trigger another inference request. The raw response is saved.
        if not isinstance(body["choices"][0]["message"].get("content"), str):
            self.outcome = (
                "provider_content_filter"
                if body["choices"][0].get("finish_reason") == "content_filter"
                else "model_output_budget"
                if body["choices"][0].get("finish_reason") == "length"
                else "model_invalid_action"
            )
            raise CapturedModelActionError("Captured model response has no textual action; no creator retry")
        self.no_resend = False
        self.outcome = (
            "model_output_budget" if body["choices"][0].get("finish_reason") == "length" else "captured_model_action"
        )
        return native

    def close(self) -> None:
        self.transport.close()


@contextmanager
def bind_creator_model(module: ModuleType, source_root: Path, transport: CreatorModelHTTP) -> Iterator[None]:
    """Bind only the exact pinned creator module in an isolated synchronous worker."""
    manifest = json.loads(MANIFEST.read_text())
    path = checked_source(source_root, "sci/base/model.py", manifest["files"]["sci/base/model.py"])
    if Path(module.__file__).resolve() != path.resolve():
        raise ValueError("Unexpected creator Model import")
    original = module.requests
    if original is not requests:
        raise ValueError("Creator model transport already overridden")
    module.requests = transport
    try:
        yield
    finally:
        try:
            transport.close()
        finally:
            module.requests = original
