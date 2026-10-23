# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal child entrypoint: only the pinned RAG module and public prompt are visible."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import httpx
from openai import OpenAI


def load_file(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate(
    public: Path, output: Path, expected: dict[str, str], *, transport: httpx.BaseTransport | None = None
) -> None:
    actual = {}
    for path in public.rglob("*"):
        if path.is_symlink():
            raise ValueError("Public generator input symlink")
        if path.is_file():
            actual[str(path.relative_to(public))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if set(expected) != {"rag.py", "prompt.txt"} or actual != expected:
        raise ValueError("Public generator input binding changed")
    if (output / "generation.json").exists():
        raise FileExistsError("Generator output already exists")
    rag = load_file("metasyn_public_rag", public / "rag.py")
    with httpx.Client(transport=transport, timeout=600, follow_redirects=False) as client:

        def factory(**kwargs):
            if set(kwargs) - {"api_key", "base_url"}:
                raise ValueError("Unexpected creator SDK options")
            return OpenAI(
                api_key="task-scoped-placeholder", base_url="http://gym-internal/v1", http_client=client, max_retries=0
            )

        with patch.object(rag, "OpenAI", factory):
            report, usage = rag.generate_report(
                (public / "prompt.txt").read_text(),
                "nvidia/nvidia/nemotron-3.5-super-vl",
                api_key="task-scoped-placeholder",
            )
    with (output / "generation.json").open("x") as handle:
        json.dump({"report": report, "usage": usage}, handle)


if __name__ == "__main__":
    shared = load_file("shared_guarded_transport", Path("/control/guarded_chat_transport.py"))
    transport = shared.GuardedSyncChatTransport(
        socket_path="/bridge/model.sock",
        token=os.environ["METASYN_BRIDGE_TOKEN"],
        lineage=os.environ["METASYN_BRIDGE_LINEAGE"],
        digest=os.environ["METASYN_BRIDGE_DIGEST"],
        namespace="metasyn",
        model="nvidia/nvidia/nemotron-3.5-super-vl",
        output=Path("/workspace/provider-calls"),
    )
    generate(Path("/public"), Path("/workspace"), json.loads(os.environ["METASYN_PUBLIC_HASHES"]), transport=transport)
