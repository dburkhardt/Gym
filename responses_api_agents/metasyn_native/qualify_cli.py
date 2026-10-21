# SPDX-License-Identifier: Apache-2.0
"""Full native CLI collection/resume control with a deterministic loopback provider."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

from aiohttp import web
import yaml

from benchmarks.metasyn.prepare import stage
from responses_api_agents.metasyn_native.app import MODEL


def response_for(prompt, expected):
    if prompt == expected:
        return "# Deterministic test report\nNo articles were included.\nAdults with diagnosed disease. Animal studies excluded. Evidence is limited."
    if "Extract criteria now:" in prompt:
        return json.dumps(
            {"inclusion_criteria": ["Adults with diagnosed disease"], "exclusion_criteria": ["Animal studies"]}
        )
    if "Extract conclusion and insights now:" in prompt:
        return json.dumps({"conclusion_direction": "Mixed", "key_insights": ["Evidence is limited."]})
    if "consistency_score" in prompt:
        return json.dumps({"consistency_score": 0.2, "reason": "Deterministic transport control"})
    if "structure quality" in prompt:
        return json.dumps({"score": 2, "reason": "Deterministic transport control"})
    raise ValueError("Unexpected creator request")


async def main(args):
    source = Path(__file__).resolve().parents[2]
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=False)
    data = args.data.absolute()
    refs = args.references.absolute()
    expected = (data / "rag/203/prompt.txt").read_text()
    assert len(expected) == 447052
    sample = stage(data, out / "benchmark.jsonl", 20)
    calls = []

    async def provider(request):
        payload = await request.json()
        assert payload["model"] == MODEL
        prompt = payload["messages"][-1]["content"]
        content = response_for(prompt, expected)
        calls.append(
            {
                "ordinal": len(calls),
                "characters": len(prompt),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )
        (out / "provider-calls.json").write_text(json.dumps(calls, indent=2))
        return web.json_response(
            {
                "id": f"fixture-{len(calls)}",
                "object": "chat.completion",
                "created": 0,
                "model": MODEL,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    app = web.Application(client_max_size=16 * 1024**2)
    app.router.add_post("/v1/chat/completions", provider)
    runner = web.AppRunner(app)
    report = {"status": "failed", "model_calls": calls}
    child = None
    try:
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        for component in (
            "responses_api_agents/metasyn_native",
            "responses_api_models/openai_model",
            "environment_servers/legacy_agent",
        ):
            dest = out / "venvs" / component / ".venv"
            dest.parent.mkdir(parents=True)
            dest.symlink_to(Path(sys.executable).absolute().parent.parent)
        config = {
            "config_paths": ["benchmarks/metasyn/config.yaml"],
            "policy_model_name": MODEL,
            "policy_api_key": "local-control-only",
            "policy_base_url": f"http://127.0.0.1:{port}/v1",
            "observability_enabled": True,
            "provider_dispatch_guard": True,
            "model_call_capture_dir": str(out / "capture"),
            "head_server": {"host": "127.0.0.1", "port": args.head_port},
            "port_range_low": args.head_port + 1,
            "port_range_high": args.head_port + 30,
            "uv_venv_dir": str(out / "venvs"),
            "skip_venv_if_present": True,
            "use_cached_prepared_benchmarks": True,
            "policy_model": {
                "responses_api_models": {
                    "openai_model": {
                        "upstream_max_num_tries": 1,
                        "max_http_attempts": 1,
                        "upstream_request_timeout_seconds": 30,
                    }
                }
            },
            "metasyn_agent": {
                "responses_api_agents": {
                    "metasyn_native": {
                        "datasets": [
                            {
                                "name": "metasyn",
                                "type": "benchmark",
                                "jsonl_fpath": str(sample),
                                "prepare_script": "benchmarks/metasyn/prepare.py",
                                "num_repeats": 1,
                            }
                        ]
                    }
                }
            },
        }
        (out / "runtime.yaml").write_text(yaml.safe_dump(config))
        env = os.environ.copy()
        env.update(
            PYTHONPATH=str(source),
            NEMO_GYM_EXTRA_ROOTS=str(source),
            METASYN_DATA=str(data),
            METASYN_REFERENCES=str(refs),
            METASYN_CREATOR_SOURCE=str(args.creator_source.absolute()),
            METASYN_PYTHON=str(args.python.absolute()),
            METASYN_RUNNER=str(source / "responses_api_agents/metasyn_native/native_runner.py"),
            METASYN_OUTPUT=str(out / "attempt"),
            METASYN_INFERENCE_ENABLED="true",
            METASYN_SCIENTIFIC_THREADS="1",
            METASYN_PREPARED_SAMPLE=str(sample),
            METASYN_SAMPLE_COUNT="20",
        )
        command = [
            sys.executable,
            "-c",
            "from nemo_gym.cli.main import main; main()",
            "eval",
            "run",
            "--config",
            str(out / "runtime.yaml"),
            "--model-type",
            "openai_model",
            "--model",
            MODEL,
            "--split",
            "benchmark",
            "--limit",
            "1",
            "--num-repeats",
            "1",
            "--concurrency",
            "1",
            "--health-check-workers",
            "1",
            "--output",
            str(out / "results.jsonl"),
        ]
        (out / "command.json").write_text(json.dumps(command, indent=2))
        results = []
        for resume in (False, True):
            with (out / ("resume.log" if resume else "run.log")).open("x") as log:
                child = await asyncio.create_subprocess_exec(
                    *(command + (["--resume"] if resume else [])),
                    cwd=source,
                    env=env,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                )
                try:
                    code = await asyncio.wait_for(child.wait(), 180)
                except asyncio.TimeoutError:
                    child.send_signal(signal.SIGINT)
                    await asyncio.wait_for(child.wait(), 30)
                    raise
            results.append({"resume": resume, "exit_code": code, "provider_calls": len(calls)})
            (out / "progress.json").write_text(json.dumps(results, indent=2))
            assert code == 0, results
            assert len(calls) == 5, results
        rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
        assert len(rows) == 1
        report = {
            "status": "passed",
            "collection": results,
            "provider_calls": calls,
            "results_rows": 1,
            "prepared_frozen_rows": 20,
            "scope": "Native Gym CLI preparation/collection, real agent run route, actual model capture and completed-output resume; deterministic provider only; no arbitrary interrupted-state recovery",
        }
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        cleanup_errors = []
        try:
            if child is not None and child.returncode is None:
                child.send_signal(signal.SIGINT)
                await asyncio.wait_for(child.wait(), 30)
        except BaseException as error:
            cleanup_errors.append(repr(error))
        try:
            await runner.cleanup()
        except BaseException as error:
            cleanup_errors.append(repr(error))
        report["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            report["status"] = "failed"
        (out / "qualification.json").write_text(json.dumps(report, indent=2))
        if cleanup_errors:
            raise RuntimeError("Native qualification cleanup incomplete; inspect owned processes")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--creator-source", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--head-port", type=int, default=28701)
    asyncio.run(main(p.parse_args()))
