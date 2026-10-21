# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact creator-lock execution; Gym networking and sandbox ownership stay in parent."""

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from benchmarks.bioagentbench.child_runtime import runtime_class


_OWNER = threading.Lock()


@asynccontextmanager
async def parent_interception(packet, endpoint, secret):
    async with httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(uds=packet["model_socket"], retries=0),
        timeout=610,
        headers={"x-bioagent-ipc-token": packet["ipc_token"]},
    ) as client:
        response = await client.post("http://parent/interception/start", json={"endpoint": endpoint, "secret": secret})
        response.raise_for_status()
        try:
            yield
        finally:
            response = await client.post("http://parent/interception/stop", json={})
            response.raise_for_status()


async def run_native(*, config: dict, packet: dict, output: Path) -> list[dict]:
    """Delegate one frozen task to native Environment/run_eval; never native resume."""
    import verifiers.v1 as vf
    import verifiers.v1.runtimes as runtimes
    from bioagent_opencode.harness import OpenCodeHarness
    from verifiers.v1 import judge
    from verifiers.v1.cli.eval import runner
    from verifiers.v1.cli.resolve import narrow_config
    from verifiers.v1.clients import config as clients
    from verifiers.v1.clients.eval import EvalClient
    from verifiers.v1.configs.eval import EvalConfig

    if config.get("resume") or config.get("num_rollouts") != 1 or config.get("max_concurrent") != 1:
        raise ValueError("Exactly one fresh creator rollout is supported")
    if config["taskset"].get("task_ids") or config["taskset"]["task_id"] != packet["task_id"]:
        raise ValueError("Frozen native task selection changed")
    if config["model"] != packet["model"] or config["taskset"]["task"]["judge"]["model"] != packet["model"]:
        raise ValueError("All native model roles must match the admitted model")
    if config["taskset"]["staging"] != "local":
        raise ValueError("Qualified public assets must be staged before native execution")
    if config.get("retries", {}).get("rollout", {}).get("max_retries", 0) != 0:
        raise ValueError("Whole-rollout retries are not permitted")
    if not output.is_dir() or (output / "native").exists() or (output / "traces.json").exists():
        raise ValueError("Creator output is not a fresh parent-owned execution directory")

    transport = httpx.AsyncHTTPTransport(uds=packet["model_socket"], retries=0)
    Managed = runtime_class(
        socket=Path(packet["runtime_socket"]),
        token=packet["ipc_token"],
        deadline=packet["deadline"],
        expected_config=packet["runtime_config"],
    )
    # Verifiers must produce a host-loopback interception URL, which launch below
    # explicitly tunnels into the isolated container. This is not host networking.
    Managed.is_local = True

    class GuardedEvalClient(EvalClient):
        def __init__(self, base_url, api_key, headers=None):
            self.base_url, self.api_key, self.headers = base_url, api_key, dict(headers or {})
            self.http = httpx.AsyncClient(
                transport=transport, timeout=600, headers={"x-bioagent-ipc-token": packet["ipc_token"]}
            )

    def sdk(config):
        return AsyncOpenAI(
            base_url="http://native/v1",
            api_key="task-local-placeholder",
            http_client=httpx.AsyncClient(
                transport=transport, timeout=600, headers={"x-bioagent-ipc-token": packet["ipc_token"]}
            ),
            max_retries=0,
        )

    original = (runtimes.DockerRuntime, clients.EvalClient, judge.build_async_openai, OpenCodeHarness.launch)
    original_launch = OpenCodeHarness.launch

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls):
        if mcp_urls:
            raise ValueError("Unqualified external MCP services are not exposed by this frozen recipe")
        async with parent_interception(packet, endpoint, secret):
            await runtime.run_background(
                [
                    "python",
                    "/gym-bridge/relay.py",
                    "--socket",
                    "/gym-bridge/intercept.sock",
                    "--port-file",
                    ".vf-relay-port.json",
                ],
                {},
                ".vf-relay.log",
            )
            async with asyncio.timeout(30):
                while True:
                    try:
                        port = json.loads(await runtime.read(".vf-relay-port.json"))["port"]
                        break
                    except (RuntimeError, ValueError, KeyError):
                        await asyncio.sleep(0.1)
            return await original_launch(self, ctx, trace, runtime, f"http://127.0.0.1:{port}/v1", secret, mcp_urls)

    if not _OWNER.acquire(blocking=False):
        raise RuntimeError("Native creator module patches already owned by another invocation")
    try:
        runtimes.DockerRuntime = Managed
        clients.EvalClient = GuardedEvalClient
        judge.build_async_openai = sdk
        OpenCodeHarness.launch = launch
        selected = narrow_config(EvalConfig, ["--taskset.id", "bioagent-taskset", "--harness.id", "bioagent-opencode"])
        resolved = selected.model_validate(
            {**config, "output_dir": str(output / "native"), "push": False, "rich": False}
        )
        traces = await runner.run_eval(vf.Environment(resolved), resolved)
        result = [trace.model_dump(mode="json") for trace in traces]
        (output / "traces.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        runtimes.DockerRuntime, clients.EvalClient, judge.build_async_openai, OpenCodeHarness.launch = original
        try:
            await transport.aclose()
        finally:
            _OWNER.release()
