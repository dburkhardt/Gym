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


class GiabMetricCapture:
    """Observe the pinned scorer result without changing its return or exception."""

    def __init__(self, original):
        self.original = original
        self.result = None
        self.called = False

    async def __call__(self, task_id, artifact_dir, source_dir):
        if task_id != "giab" or self.called:
            raise ValueError("GIAB metric capture requires one admitted GIAB rollout")
        self.called = True
        result = await self.original(task_id, artifact_dir, source_dir)
        self.result = json.loads(json.dumps(result))
        return result

    def attach(self, traces):
        if len(traces) != 1:
            raise ValueError("GIAB metric capture requires exactly one native trace")
        if self.result is not None:
            info = traces[0].setdefault("info", {})
            if "gym_giab_benchmark_metrics" in info:
                raise ValueError("Reserved GIAB metric observation already exists")
            info["gym_giab_benchmark_metrics"] = self.result


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


async def relay_port(runtime) -> int:
    """Wait for an atomically published relay file without retrying failed IPC."""
    async with asyncio.timeout(30):
        while True:
            ready = await runtime.run(["test", "-s", ".vf-relay-port.json"], {})
            if ready.exit_code == 0:
                break
            if ready.exit_code != 1:
                raise RuntimeError("Relay readiness command failed")
            await asyncio.sleep(0.1)
        port = json.loads(await runtime.read(".vf-relay-port.json"))["port"]
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Invalid relay port")
        return port


async def run_native(*, config: dict, packet: dict, output: Path) -> list[dict]:
    """Delegate one frozen task to native Environment/run_eval; never native resume."""
    import bioagent_taskset.taskset as creator_taskset
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
    original_metrics = creator_taskset.benchmark_metrics
    metric_capture = GiabMetricCapture(original_metrics) if packet["task_id"] == "giab" else None

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
            port = await relay_port(runtime)
            return await original_launch(self, ctx, trace, runtime, f"http://127.0.0.1:{port}/v1", secret, mcp_urls)

    if not _OWNER.acquire(blocking=False):
        raise RuntimeError("Native creator module patches already owned by another invocation")
    try:
        if metric_capture is not None:
            creator_taskset.benchmark_metrics = metric_capture
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
        if metric_capture is not None:
            metric_capture.attach(result)
        (output / "traces.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        creator_taskset.benchmark_metrics = original_metrics
        runtimes.DockerRuntime, clients.EvalClient, judge.build_async_openai, OpenCodeHarness.launch = original
        try:
            await transport.aclose()
        finally:
            _OWNER.release()
