# SPDX-License-Identifier: Apache-2.0
"""Actual native CLI control with the pinned creator and a local provider only."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import yaml
from aiohttp import web

from benchmarks.bioagentbench.prepare import rows
from benchmarks.bioagentbench.qualify_runtime import MODEL, fixture


def fixture_response(payload, ordinal):
    role, response = fixture(payload)
    response["id"] = f"chatcmpl-bioagent-native-fixture-{ordinal}"
    return role, response


def stream_chunks(response):
    return [
        {
            **response,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": response["choices"][0]["message"], "finish_reason": None}],
        },
        {
            **response,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]


async def qualify(args):
    root = Path(__file__).resolve().parents[2]
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=False)
    packet = json.loads(args.packet.read_text())
    native = packet["native_config"]
    selected = [r for r in rows(Path(native["taskset"]["metadata_path"])) if r["task_id"] == "cystic-fibrosis"]
    assert len(selected) == 1
    sample = out / "sample.jsonl"
    sample.write_text(json.dumps(selected[0]) + "\n")
    calls = []

    async def provider(request):
        payload = await request.json()
        if len(calls) >= 12:
            raise web.HTTPBadRequest(text="Fixture request bound reached")
        role, response = fixture_response(payload, len(calls))
        calls.append({"role": role, "request": payload, "response": response})
        (out / "fixture-calls.json").write_text(json.dumps(calls, indent=2))
        if not payload.get("stream"):
            return web.json_response(response)
        stream = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await stream.prepare(request)
        for chunk in stream_chunks(response):
            await stream.write(("data: " + json.dumps(chunk) + "\n\n").encode())
        await stream.write(b"data: [DONE]\n\n")
        await stream.write_eof()
        return stream

    app = web.Application(client_max_size=16 * 1024**2)
    app.router.add_post("/v1/chat/completions", provider)
    runner = web.AppRunner(app)
    await runner.setup()
    child = None
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        for component in (
            "responses_api_agents/bioagent_native",
            "responses_api_models/openai_model",
            "environment_servers/legacy_agent",
        ):
            dest = out / "venvs" / component / ".venv"
            dest.parent.mkdir(parents=True)
            dest.symlink_to(Path(sys.executable).absolute().parent.parent)
        config = {
            "config_paths": ["responses_api_agents/bioagent_native/configs/bioagent.yaml"],
            "policy_model_name": MODEL,
            "policy_api_key": "local-fixture-only",
            "policy_base_url": f"http://127.0.0.1:{port}/v1",
            "provider_dispatch_guard": True,
            "observability_enabled": True,
            "model_call_capture_dir": str(out / "capture"),
            "head_server": {"host": "127.0.0.1", "port": args.head_port},
            "port_range_low": args.head_port + 1,
            "port_range_high": args.head_port + 30,
            "uv_venv_dir": str(out / "venvs"),
            "skip_venv_if_present": True,
            "use_cached_prepared_benchmarks": True,
            "dispatch_budget_s": 27000,
            "drain_margin_s": 25980,
            "policy_model": {
                "responses_api_models": {
                    "openai_model": {
                        "max_concurrent_requests": 1,
                        "upstream_request_timeout_seconds": 600,
                        "upstream_max_num_tries": 1,
                        "upstream_retry_policy": {"max_attempts": 1},
                    }
                }
            },
            "bioagent_agent": {
                "responses_api_agents": {
                    "bioagent_native": {
                        "datasets": [
                            {
                                "name": "bioagentbench",
                                "type": "benchmark",
                                "jsonl_fpath": str(sample),
                                "prepare_script": "benchmarks/bioagentbench/prepare.py",
                                "num_repeats": 1,
                            }
                        ]
                    }
                }
            },
        }
        config_path = out / "runtime.yaml"
        config_path.write_text(yaml.safe_dump(config))
        env = {
            **os.environ,
            "PYTHONPATH": str(root),
            "NEMO_GYM_EXTRA_ROOTS": str(root),
            "NEMO_GYM_SERVICE_OWNERSHIP_MODE": "systemd",
            "BIOAGENT_NATIVE_PYTHON": packet["interpreter"]["path"],
            "BIOAGENT_CREATOR_ROOT": packet["creator_root"],
            "BIOAGENT_METADATA": native["taskset"]["metadata_path"],
            "BIOAGENT_ASSET_ROOT": native["taskset"]["data_root"],
            "BIOAGENT_ASSET_MANIFEST": packet["asset_manifest"],
            "BIOAGENT_IMAGE": packet["image"],
            "BIOAGENT_OUTPUT_ROOT": str(out / "artifacts"),
        }
        command = [
            sys.executable,
            "-c",
            "from nemo_gym.cli.main import main; main()",
            "eval",
            "run",
            "--model-type",
            "openai_model",
            "--config",
            str(config_path),
            "--model",
            MODEL,
            "--split",
            "benchmark",
            "--limit",
            "1",
            "--num-repeats",
            "1",
            "--temperature",
            "0.2",
            "--max-output-tokens",
            "32768",
            "--concurrency",
            "1",
            "--output",
            str(out / "results.jsonl"),
        ]
        for name, extra in (("plan", ["--plan-only"]), ("run", [])):
            with (out / f"{name}.log").open("wb") as log:
                child = await asyncio.create_subprocess_exec(
                    *command, *extra, cwd=root, env=env, stdout=log, stderr=asyncio.subprocess.STDOUT
                )
                (out / f"{name}-pid.json").write_text(json.dumps({"pid": child.pid}))
                code = await child.wait()
                (out / f"{name}-exit.json").write_text(json.dumps({"returncode": code}))
                if code:
                    raise RuntimeError(f"Native {name} failed with exit {code}; preserve attempt")
            if name == "plan" and calls:
                raise AssertionError("Plan performed model requests")
        results = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
        assert len(results) == 1 and results[0]["reward"] == 0
        assert {call["role"] for call in calls} == {"solver", "judge"}
        (out / "complete.json").write_text(
            json.dumps(
                {
                    "fixture_calls": len(calls),
                    "external_model_calls": 0,
                    "qualification": "Native CLI and actual creator/Docker with local provider; not model performance. Capture and cleanup require independent reconciliation.",
                },
                indent=2,
            )
        )
    finally:
        if child is not None and child.returncode is None:
            child.terminate()
            try:
                await asyncio.wait_for(child.wait(), 30)
            except asyncio.TimeoutError:
                child.kill()
                await child.wait()
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-port", type=int, default=24000)
    asyncio.run(qualify(parser.parse_args()))


if __name__ == "__main__":
    main()
