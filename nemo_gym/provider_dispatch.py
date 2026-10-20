# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-local external dispatch evidence, persisted by the existing capture store.

An intent is deliberately conservative: it does not prove that a provider accepted
anything. Once intent is durable, a crash must be treated as ambiguous until a
complete response has been durably captured or the provider can reconcile it.
"""

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable
from uuid import uuid4


@dataclass
class ProviderDispatch:
    """Track physical attempts inside one logical captured model call."""

    persist: Callable[[dict], None]
    attempts: list[dict] = field(default_factory=list)

    def before_send(self) -> None:
        if self.attempts and self.attempts[-1]["outcome"] != "rejected":
            raise RuntimeError("previous provider dispatch is unresolved; automatic resend refused")
        self.attempts.append({"attempt_id": uuid4().hex, "outcome": "unknown"})
        self.persist(self.evidence())

    def response_headers(self, status: int) -> None:
        self.attempts[-1].update(outcome="response_headers", status_code=status)
        self.persist(self.evidence())

    def evidence(self) -> dict:
        return {"provider_attempts": [dict(attempt) for attempt in self.attempts]}


current_provider_dispatch: ContextVar[ProviderDispatch | None] = ContextVar("provider_dispatch", default=None)


def configuration_digest(value: dict) -> str:
    """Hash configuration deterministically while excluding credential fields."""
    import hashlib
    import json

    secret_fields = {
        "api_key",
        "openai_api_key",
        "anthropic_api_key",
        "auth_token",
        "control_auth_token",
        "authorization",
        "access_token",
        "refresh_token",
        "bearer_token",
        "password",
    }

    def clean(item: object) -> object:
        if isinstance(item, dict):
            return {key: clean(child) for key, child in item.items() if key.lower() not in secret_fields}
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        return item

    local_server_fields = {
        "host",
        "port",
        "num_workers",
        "uvicorn_proxy_headers",
        "uvicorn_forwarded_allow_ips",
    }
    semantic = {key: item for key, item in value.items() if key not in local_server_fields}
    return hashlib.sha256(json.dumps(clean(semantic), sort_keys=True, default=str).encode()).hexdigest()
