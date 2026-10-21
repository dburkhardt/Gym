# SPDX-License-Identifier: Apache-2.0
"""Bind the creator ReAct transport to native Gym capture without changing its loop."""

import asyncio
import hashlib
import json
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit


_OWNER = threading.Lock()

MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


def publish(path: Path, value: dict):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def provider_messages(messages: list[dict]) -> list[dict]:
    """Remove creator response IDs, not scientific content or tool call IDs."""
    return [
        {key: value for key, value in message.items() if not (message.get("role") == "assistant" and key == "id")}
        for message in messages
    ]


@contextmanager
def guarded_agent(
    endpoint: str,
    lineage: str,
    calls: Path,
    *,
    client=None,
    dispatch_digest: str | None = None,
    continuation_session=None,
    capture_dir: Path | None = None,
    episode=None,
    deadline: float | None = None,
    before_dispatch=None,
):
    from corral.agents import base_agent
    from corral.cli import create_agent

    if "/ng-rollout/" not in endpoint or not endpoint.endswith("/v1"):
        raise ValueError("Corral requires a native rollout-qualified model service")
    if dispatch_digest is not None:
        if len(dispatch_digest) != 64 or any(c not in "0123456789abcdef" for c in dispatch_digest):
            raise ValueError("Invalid persisted Corral protocol digest")
        digest = dispatch_digest
    else:
        from nemo_gym.provider_dispatch import configuration_digest

        digest = configuration_digest(
            {
                "model": MODEL,
                "temperature": 1.0,
                "max_tokens": 16384,
                "iterations": 50,
                "creator": "f5a6963a9ccf6e3c0058c776fe4111b639add648",
                "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
        )
    restored = None
    if continuation_session is not None:
        from benchmarks.corral.continuation import validate_boundary

        if capture_dir is None or episode is None or deadline is None:
            raise ValueError("Corral continuation requires original capture and deadline binding")
        parsed = urlsplit(endpoint)
        route = re.fullmatch(r"/ng-rollout/([^/]+)/v1", parsed.path)
        if route is None or route.group(1) != episode.capture_key or parsed.query or parsed.fragment:
            raise ValueError("Corral continuation endpoint differs from original capture identity")
        restored = validate_boundary(
            continuation_session,
            calls=calls,
            lineage=lineage,
            protocol=digest,
            capture_dir=capture_dir,
            episode=episode,
            deadline=deadline,
        )
    else:
        calls.mkdir(parents=True, exist_ok=False)
    original = base_agent.llm_call
    lock = asyncio.Lock()
    ordinal = 0 if restored is None else restored["next_ordinal"]
    failed = False

    async def call(**kwargs):
        nonlocal ordinal, failed
        async with lock:
            if failed:
                raise RuntimeError("Corral call unresolved; new dispatch forbidden")
            if kwargs["model"] != "openai/" + MODEL or kwargs["api_endpoint"] != endpoint:
                raise ValueError("Creator model or endpoint changed")
            if kwargs.get("max_tokens") != 16384 or kwargs.get("temperature") != 1.0:
                raise ValueError("Creator generation settings changed")
            if "messages" in kwargs:
                kwargs["messages"] = provider_messages(kwargs["messages"])
            if before_dispatch is not None:
                # Internal controller hook, before intent or external dispatch. The
                # creator's own local history reaches this point without reconstruction.
                await before_dispatch(kwargs["messages"], ordinal)
            if ordinal >= 50:
                raise RuntimeError("Corral original provider budget exhausted")
            stem = calls / f"{ordinal:06d}"
            headers = dict(kwargs.pop("extra_headers", {}) or {})
            headers.update(
                {"x-nemo-gym-logical-call-id": f"corral:{lineage}:{ordinal}", "x-nemo-gym-call-config-sha256": digest}
            )
            publish(
                stem.with_suffix(".intent.json"),
                {"logical_call": headers["x-nemo-gym-logical-call-id"], "config_digest": digest},
            )
            failed = True
            try:
                result = await original(**kwargs, extra_headers=headers)
            except BaseException as error:
                publish(stem.with_suffix(".failure.json"), {"type": type(error).__name__, "resend": False})
                raise
            # Raw provider response remains durably captured by the Gym model service.
            publish(stem.with_suffix(".complete.json"), {"completed": True})
            ordinal += 1
            failed = False
            return result

    if not _OWNER.acquire(blocking=False):
        raise RuntimeError("Another Corral native transport owns this process")
    base_agent.llm_call = call
    try:
        agent = create_agent(
            "react",
            model="openai/" + MODEL,
            api_endpoint=endpoint,
            temperature=1.0,
            agent_kwargs={
                "api_key": "scoped-no-provider-secret",
                "max_tokens": 16384,
                "num_retries": 0,
                "timeout": 600,
                "client": client,
            },
        )
        if restored is not None:
            from benchmarks.corral.continuation import restored_agent

            agent = restored_agent(agent, restored)
        yield agent
    finally:
        base_agent.llm_call = original
        _OWNER.release()
