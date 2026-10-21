# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run creator Verifiers unchanged, replacing only owned runtime and transport."""

import asyncio
import hmac
import json
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from aiohttp import web
from openai import AsyncOpenAI

from benchmarks.bioagentbench.runtime import runtime_class
from benchmarks.bioagentbench.transport import NativeTransport
from nemo_gym.sandbox.model_bridge import _settle
from nemo_gym.server_utils import request


_OWNER = threading.Lock()


@asynccontextmanager
async def interception_relay(*, endpoint: str, secret: str, socket_path: Path):
    """Forward only this rollout's authenticated creator interception route."""
    target = urlsplit(endpoint)
    if target.scheme != "http" or target.hostname not in ("127.0.0.1", "localhost") or target.query or target.fragment:
        raise ValueError("Creator interception must be a loopback-only endpoint")
    app = web.Application(client_max_size=16 * 1024 * 1024)

    async def forward(incoming):
        if not hmac.compare_digest(incoming.headers.get("Authorization", ""), "Bearer " + secret):
            raise web.HTTPUnauthorized()
        payload = await incoming.read()
        upstream = await request(
            "POST",
            endpoint.rstrip("/") + "/chat/completions",
            data=payload,
            headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json"},
            allow_redirects=False,
            _max_num_tries=1,
            _max_connection_retries=0,
            _resend_after_disconnect=False,
        )
        try:
            result = web.StreamResponse(
                status=upstream.status,
                headers={"Content-Type": upstream.headers.get("Content-Type", "application/json")},
            )
            await result.prepare(incoming)
            async for chunk in upstream.content.iter_chunked(65536):
                await result.write(chunk)
            await result.write_eof()
            return result
        finally:
            upstream.release()

    app.router.add_post("/v1/chat/completions", forward)
    runner = web.AppRunner(app, shutdown_timeout=610)
    await runner.setup()
    try:
        await web.UnixSite(runner, str(socket_path)).start()
        yield
    finally:
        await _settle(asyncio.create_task(runner.cleanup()))
        socket_path.unlink(missing_ok=True)


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
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="bioagent-", dir="/tmp") as temporary:
        bridge = Path(temporary)
        shutil.copyfile(Path(__file__).with_name("relay.py"), bridge / "relay.py")
        workspace = output / "workspace"
        workspace.mkdir()
        transport = NativeTransport(
            endpoint=packet["endpoint"],
            lineage=packet["lineage"],
            digest=packet["digest"],
            model=packet["model"],
            output=output / "calls",
        )
        Managed = runtime_class(
            workspace=workspace,
            source_dir=Path(config["taskset"]["data_root"]) / packet["task_id"],
            receipt=output / "sandbox-lifecycle.json",
            image=packet["image"],
            bridge_dir=bridge,
        )
        # Verifiers must produce a host-loopback interception URL, which launch below
        # explicitly tunnels into the isolated container. This is not host networking.
        Managed.is_local = True

        class GuardedEvalClient(EvalClient):
            def __init__(self, base_url, api_key, headers=None):
                self.base_url, self.api_key, self.headers = base_url, api_key, dict(headers or {})
                self.http = httpx.AsyncClient(transport=transport, timeout=600)

        def sdk(config):
            return AsyncOpenAI(
                base_url="http://native/v1",
                api_key="task-local-placeholder",
                http_client=httpx.AsyncClient(transport=transport, timeout=600),
                max_retries=0,
            )

        original = (runtimes.DockerRuntime, clients.EvalClient, judge.build_async_openai, OpenCodeHarness.launch)
        original_launch = OpenCodeHarness.launch

        async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls):
            if mcp_urls:
                raise ValueError("Unqualified external MCP services are not exposed by this frozen recipe")
            async with interception_relay(endpoint=endpoint, secret=secret, socket_path=bridge / "intercept.sock"):
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
                return await original_launch(
                    self, ctx, trace, runtime, f"http://127.0.0.1:{port}/v1", secret, mcp_urls
                )

        if not _OWNER.acquire(blocking=False):
            raise RuntimeError("Native creator module patches already owned by another invocation")
        try:
            runtimes.DockerRuntime = Managed
            clients.EvalClient = GuardedEvalClient
            judge.build_async_openai = sdk
            OpenCodeHarness.launch = launch
            selected = narrow_config(
                EvalConfig, ["--taskset.id", "bioagent-taskset", "--harness.id", "bioagent-opencode"]
            )
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
