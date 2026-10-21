# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run full native creator phases with a local deterministic provider, never an LLM."""

import argparse
import asyncio
import hashlib
import json
import shlex
import time
from pathlib import Path

from aiohttp import web

from benchmarks.bioagentbench.outcomes import disposition
from nemo_gym.sandbox.guarded_chat_transport import publish
from nemo_gym.sandbox.providers.local.provider import LocalProvider
from nemo_gym.sandbox.scoring_workspace import scoring_workspace


MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


def fixture(payload: dict) -> tuple[str, dict]:
    """A no-answer solver control and typed creator judge response, not scientific performance."""
    if payload.get("model") != MODEL:
        raise ValueError("Unexpected model")
    judge = payload.get("response_format", {}).get("type") == "json_schema"
    role = "judge" if judge else "solver"
    content = (
        json.dumps(
            {
                "steps_completed": 0,
                "steps_to_completion": 1,
                "final_result_reached": False,
                "notes": "Deterministic no-output infrastructure control.",
            }
        )
        if judge
        else "No scientific output was produced in this deterministic infrastructure control."
    )
    return role, {
        "id": "chatcmpl-offline-control",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
    }


async def qualify(packet: dict, directory: Path) -> None:
    """Preserve full inputs and native 8 GiB runtime; reject altered control protocols."""
    native = packet["native_config"]
    if native["harness"]["runtime"]["memory"] != 8 or native["harness"]["runtime"]["cpu"] != 2:
        raise ValueError("Native runtime resources changed")
    if native["timeout"] != {"setup": 3600, "rollout": 14400, "finalize": 3600, "scoring": 3600}:
        raise ValueError("Native phase budgets changed")
    if packet["task_id"] != "cystic-fibrosis":
        raise ValueError("Initial complete no-output control is scoped to original cystic-fibrosis task")
    directory.mkdir(parents=True, exist_ok=False)
    calls = []

    async def provider(request):
        payload = await request.json()
        if (
            len(calls) >= 12
            or not request.headers.get("x-nemo-gym-logical-call-id")
            or not request.headers.get("x-nemo-gym-call-config-sha256")
        ):
            raise web.HTTPBadRequest()
        role, response = fixture(payload)
        calls.append(
            {
                "role": role,
                "request": payload,
                "response": response,
                "logical_call_id": request.headers["x-nemo-gym-logical-call-id"],
            }
        )
        publish(directory / "fixture-calls.json", {"calls": calls, "live_provider_calls": 0})
        if not payload.get("stream"):
            return web.json_response(response)
        stream = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await stream.prepare(request)
        chunk = {
            **response,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": response["choices"][0]["message"], "finish_reason": None}],
        }
        await stream.write(("data: " + json.dumps(chunk) + "\n\n").encode())
        chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        await stream.write(("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())
        await stream.write_eof()
        return stream

    app = web.Application()
    app.router.add_post("/ng-rollout/bioagent-control/v1/chat/completions", provider)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        packet["endpoint"] = (
            f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/ng-rollout/bioagent-control/v1"
        )
        packet["lineage"] = "bioagent-control-" + directory.name
        native["taskset"]["task"]["artifact_root"] = str(directory / "artifacts")
        packet["digest"] = hashlib.sha256(json.dumps(packet, sort_keys=True).encode()).hexdigest()
        publish(directory / "packet.json", packet)
        local = LocalProvider(concurrency=1, workspace_root="/tmp")
        async with scoring_workspace(local) as handle:
            execution = await local.exec(
                handle,
                shlex.join(
                    [
                        packet["interpreter"]["path"],
                        "-m",
                        "benchmarks.bioagentbench.worker",
                        "--packet",
                        str(directory / "packet.json"),
                    ]
                ),
                timeout_s=25980,
                terminate_background_children=True,
            )
        publish(
            directory / "execution.json",
            {
                "return_code": execution.return_code,
                "error_type": execution.error_type,
                "stdout": execution.stdout,
                "stderr": execution.stderr,
            },
        )
        if execution.return_code or execution.error_type:
            raise RuntimeError("Actual native creator control failed; preserved without retry")
        traces = json.loads((directory / "result.json").read_text())["traces"]
        if (
            len(traces) != 1
            or not disposition(traces[0])["measurement_available"]
            or traces[0]["rewards"]["results_match"] != 0
        ):
            raise AssertionError("Native no-output control did not finish scoring as expected")
        if {call["role"] for call in calls} != {"solver", "judge"}:
            raise AssertionError("Both actual creator model paths were not exercised")
        publish(
            directory / "complete.json",
            {
                "live_provider_calls": 0,
                "native_trace_disposition": disposition(traces[0]),
                "qualification": "Actual native creator/OpenCode/Docker with deterministic responses; not model performance or native Gym provider capture.",
            },
        )
    finally:
        await runner.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packet", type=Path, required=True, help="Qualified full frozen-input packet from maintained native agent"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(qualify(json.loads(args.packet.read_text()), args.output.absolute()))


if __name__ == "__main__":
    main()
