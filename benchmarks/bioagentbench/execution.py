# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run creator Verifiers unchanged, replacing only owned runtime and transport."""

import asyncio
import hashlib
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

from benchmarks.bioagentbench.runtime import runtime_class
from benchmarks.bioagentbench.scorer_environment import child_environment
from benchmarks.bioagentbench.transport import NativeDispatchStopped, NativeTransport
from nemo_gym.sandbox.model_bridge import _settle
from nemo_gym.server_utils import request


_OWNER = threading.Lock()


async def relay_model_request(incoming: web.Request, transport: NativeTransport) -> web.StreamResponse:
    """Return terminal local admission failures without encouraging creator SDK retries."""
    try:
        response = await transport.handle_async_request(
            httpx.Request("POST", "http://native/v1/chat/completions", content=await incoming.read())
        )
    except NativeDispatchStopped as error:
        # 429/5xx invite automatic OpenCode retries. This is a local terminal
        # decision; the original provider status/body remain in durable receipts.
        return web.json_response(
            {
                "error": {
                    "message": str(error),
                    "type": "native_dispatch_stopped",
                    "upstream_status": error.upstream_status,
                }
            },
            status=422,
        )
    try:
        downstream = web.StreamResponse(
            status=response.status_code,
            headers={"Content-Type": response.headers.get("content-type", "application/json")},
        )
        await downstream.prepare(incoming)
        async for chunk in response.aiter_raw():
            await downstream.write(chunk)
        await downstream.write_eof()
        return downstream
    finally:
        await response.aclose()


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
    """Parent owns one sandbox and bridge; exact creator lock runs in a child."""
    import os
    import secrets
    import shlex
    import time
    from types import SimpleNamespace

    from benchmarks.bioagentbench.runtime_channel import RuntimeServer
    from nemo_gym.sandbox.guarded_chat_transport import publish
    from nemo_gym.sandbox.providers.local.provider import LocalProvider
    from nemo_gym.sandbox.scoring_workspace import scoring_workspace

    if config.get("resume") or config.get("num_rollouts") != 1 or config.get("max_concurrent") != 1:
        raise ValueError("Exactly one fresh creator rollout is supported")
    if config["taskset"]["task_id"] != packet["task_id"] or config["model"] != packet["model"]:
        raise ValueError("Creator task/model binding changed")
    if config["taskset"]["task"]["judge"]["model"] != packet["model"]:
        raise ValueError("Creator judge model changed")
    if config["timeout"] != {"setup": 3600, "rollout": 14400, "finalize": 3600, "scoring": 3600}:
        raise ValueError("Creator phase budgets changed")
    runtime = {
        "type": "docker",
        "image": packet["image"],
        "workdir": "/app",
        "cpu": 2,
        "memory": 8,
        "gpu": None,
        "disk": None,
    }
    if any(runtime.get(k) != v for k, v in config["harness"]["runtime"].items()):
        raise ValueError("Unqualified native runtime override")
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.time() + 25980
    with tempfile.TemporaryDirectory(prefix="bioagent-parent-", dir="/tmp") as temporary:
        bridge = Path(temporary)
        os.chmod(bridge, 0o700)
        public_bridge = bridge / "container"
        public_bridge.mkdir()
        shutil.copyfile(Path(__file__).with_name("relay.py"), public_bridge / "relay.py")
        workspace = output / "workspace"
        workspace.mkdir(exist_ok=False)

        class Base:
            def __init__(self, name):
                self.name = name

        class Config(SimpleNamespace):
            def model_dump(self):
                return vars(self)

        Managed = runtime_class(
            workspace=workspace,
            source_dir=Path(config["taskset"]["data_root"]) / packet["task_id"],
            receipt=output / "sandbox-lifecycle.json",
            image=packet["image"],
            bridge_dir=public_bridge,
            runtime_types=(Base, SimpleNamespace, SimpleNamespace),
        )
        backend = Managed(
            Config(**runtime), name="bioagent-" + hashlib.sha256(packet["lineage"].encode()).hexdigest()[:24]
        )
        token = secrets.token_hex(32)
        operations = RuntimeServer(
            backend,
            socket=bridge / "runtime.sock",
            token=token,
            deadline=deadline,
            receipts=output.parent / "runtime-operations",
        )
        transport = NativeTransport(
            endpoint=packet["endpoint"],
            lineage=packet["lineage"],
            digest=packet["digest"],
            model=packet["model"],
            output=output / "calls",
        )
        interception = None
        interception_used = False
        app = web.Application(client_max_size=16 * 1024 * 1024)

        async def authenticated(incoming):
            if not hmac.compare_digest(incoming.headers.get("x-bioagent-ipc-token", ""), token):
                raise web.HTTPUnauthorized()

        async def model(incoming):
            await authenticated(incoming)
            return await relay_model_request(incoming, transport)

        async def start_interception(incoming):
            nonlocal interception, interception_used
            await authenticated(incoming)
            if interception_used:
                raise web.HTTPConflict(text="Interception replay forbidden")
            body = await incoming.json()
            if set(body) != {"endpoint", "secret"}:
                raise web.HTTPBadRequest()
            interception_used = True
            interception = interception_relay(
                endpoint=body["endpoint"], secret=body["secret"], socket_path=public_bridge / "intercept.sock"
            )
            await interception.__aenter__()
            return web.json_response({"started": True})

        async def stop_interception(incoming):
            nonlocal interception
            await authenticated(incoming)
            if interception is None:
                raise web.HTTPConflict(text="No owned interception")
            owned, interception = interception, None
            await owned.__aexit__(None, None, None)
            return web.json_response({"stopped": True})

        app.router.add_post("/v1/chat/completions", model)
        app.router.add_post("/interception/start", start_interception)
        app.router.add_post("/interception/stop", stop_interception)
        runner = web.AppRunner(app, shutdown_timeout=610)
        original_error = None
        try:
            await operations.start()
            await runner.setup()
            await web.UnixSite(runner, str(bridge / "model.sock")).start()
            child = {
                **packet,
                "native_config": config,
                "runtime_config": runtime,
                "runtime_socket": str(bridge / "runtime.sock"),
                "model_socket": str(bridge / "model.sock"),
                "ipc_token": token,
                "deadline": deadline,
                "execution_output": str(output),
                "receipt_root": str(output.parent),
            }
            child_path = bridge / "creator-packet.json"
            publish(child_path, child)
            os.chmod(child_path, 0o600)
            provider = LocalProvider(concurrency=1, workspace_root="/tmp")
            async with scoring_workspace(provider) as handle:
                command = shlex.join(
                    [
                        "env",
                        "-i",
                        *[
                            key + "=" + value
                            for key, value in child_environment(
                                packet, home=output.parent, source=Path(__file__).resolve().parents[2]
                            ).items()
                        ],
                        "BIOAGENT_LOCAL_KEY=task-local-placeholder",
                        packet["interpreter"]["path"],
                        "-m",
                        "benchmarks.bioagentbench.creator_worker",
                        "--packet",
                        str(child_path),
                    ]
                )
                command += " > " + shlex.quote(str(output.parent / "creator.log")) + " 2>&1"
                result = await provider.exec(
                    handle,
                    command,
                    timeout_s=max(0, deadline - time.time()),
                    terminate_background_children=True,
                    env={
                        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                        "BIOAGENT_LOCAL_KEY": "task-local-placeholder",
                    },
                )
                if result.error_type or result.return_code:
                    raise RuntimeError("Creator process failed; not a scientific zero")
            return json.loads((output.parent / "creator-result.json").read_text())["traces"]
        except BaseException as error:
            original_error = error
            raise
        finally:

            async def cleanup():
                try:
                    await operations.close()
                finally:
                    try:
                        if interception is not None:
                            await interception.__aexit__(None, None, None)
                    finally:
                        try:
                            await runner.cleanup()
                        finally:
                            await transport.aclose()

            try:
                await _settle(asyncio.create_task(cleanup()))
            except BaseException as cleanup_error:
                if original_error is None:
                    raise
                original_error.add_note(f"Parent owned cleanup also failed: {cleanup_error}")
