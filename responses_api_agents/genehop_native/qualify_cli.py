# SPDX-License-Identifier: Apache-2.0
"""Native CLI control using saved answers and verified cached NCBI header bytes."""

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import sys
from pathlib import Path

import yaml
from aiohttp import web

from benchmarks.genehop.native_loop import MODEL
from benchmarks.genehop.prepare import prepare
from benchmarks.genehop.source_identity import validate_source


async def main(args):
    source = Path(__file__).resolve().parents[2]
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=False)
    validate_source(args.creator_source)
    original = [json.loads(line) for line in args.sample.read_text().splitlines()]
    assert len(original) == len({row["uuid"] for row in original}) == 5
    attempt = out / "attempt"
    attempt.mkdir()
    for name in ("prompt-header.txt", "prompt-header.provenance.json"):
        shutil.copy2(args.header_evidence / name, attempt / name)
    shutil.copytree(args.header_evidence / "header-evidence", attempt / "header-evidence")
    os.environ["GENEHOP_SAMPLE"] = str(args.sample.absolute())
    os.environ["GENEHOP_PREPARED_SAMPLE"] = str(out / "benchmark.jsonl")
    sample = prepare()
    header = (attempt / "prompt-header.txt").read_text()
    expected = {}
    for row in original:
        prompt = (header + "Question: " + row["verifier_metadata"]["question"] + "\n")[-36000:]
        task = row["uuid"].replace(":", "-").replace(" ", "_")
        answer = json.loads((args.header_evidence / task / "native-result.json").read_text())["answer"]
        assert isinstance(answer, str) and not re.findall(r"\[(https?://[^\[\]]+)\]", answer)
        expected[prompt] = answer
    calls = []

    async def provider(request):
        payload = await request.json()
        assert payload["model"] == MODEL
        answer = expected[payload["messages"][-1]["content"]]
        calls.append(payload)
        (out / "provider-calls.json").write_text(json.dumps(calls, indent=2))
        return web.json_response(
            {
                "id": f"fixture-{len(calls)}",
                "object": "chat.completion",
                "created": 0,
                "model": MODEL,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    app = web.Application(client_max_size=16 * 1024**2)
    app.router.add_post("/v1/chat/completions", provider)
    runner = web.AppRunner(app)
    child = None
    report = {
        "status": "failed",
        "scope": "Saved-answer and cached-header control; no live inference, no new NCBI retrieval, not model performance or interrupted-loop recovery.",
    }
    try:
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        for component in (
            "responses_api_agents/genehop_native",
            "responses_api_models/openai_model",
            "environment_servers/legacy_agent",
        ):
            dest = out / "venvs" / component / ".venv"
            dest.parent.mkdir(parents=True)
            dest.symlink_to(Path(sys.executable).absolute().parent.parent)
        config = {
            "config_paths": ["benchmarks/genehop/config.yaml"],
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
            "genehop_agent": {
                "responses_api_agents": {
                    "genehop_native": {
                        "datasets": [
                            {
                                "name": "genehop",
                                "type": "benchmark",
                                "jsonl_fpath": str(sample),
                                "prepare_script": "benchmarks/genehop/prepare.py",
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
            GENEHOP_SOURCE=str(args.creator_source.absolute()),
            GENEHOP_OUTPUT=str(attempt),
            GENEHOP_SAMPLE=str(args.sample.absolute()),
            GENEHOP_PREPARED_SAMPLE=str(sample),
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
            "5",
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
        progress = []
        for resume in (False, True):
            with (out / ("resume.log" if resume else "run.log")).open("x") as log:
                child = await asyncio.create_subprocess_exec(
                    *(command + (["--resume"] if resume else [])),
                    cwd=source,
                    env=env,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                )
                code = await asyncio.wait_for(child.wait(), 180)
            progress.append({"resume": resume, "exit_code": code, "provider_calls": len(calls)})
            (out / "progress.json").write_text(json.dumps(progress, indent=2))
            assert code == 0 and len(calls) == 5, progress
        rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
        assert len(rows) == 5 and {row["uuid"] for row in rows} == {row["uuid"] for row in original}
        assert sum(row["raw_creator_score"] is None for row in rows) == 3
        assert all("_ng_task_index" in row and "_ng_rollout_index" in row for row in rows)
        captures = [
            json.loads(line)
            for path in (out / "capture").glob("*.capture.jsonl")
            for line in path.read_text().splitlines()
        ]
        physical = [item for row in captures for item in row.get("provider_attempts", [])]
        assert len(physical) == 5 and all(item["outcome"] == "captured" for item in physical)
        quality = json.loads((out / "quality_summary.json").read_text())["run"]
        assert quality["verdicts"] == {"healthy": 3, "unhealthy": 2, "unobserved": 0}
        assert quality["issues"]["agent_turn_hollow"] == 2
        assert quality["issues"]["trajectory_capture_mismatch"] == 0
        assert quality["issues"]["rollout_token_count_mismatch"] == 0
        assert all(len(row["ng_trajectory"]["turns"]) == len(row["ng_trajectory"]["model_calls"]) == 1 for row in rows)
        report.update(
            status="passed",
            collection=progress,
            rows=5,
            manual_only_rows=3,
            captured_physical_calls=5,
            health_verdicts=quality["verdicts"],
            hollow_historical_answers=2,
        )
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        errors = []
        try:
            if child is not None and child.returncode is None:
                child.send_signal(signal.SIGINT)
                await asyncio.wait_for(child.wait(), 30)
        except BaseException as exc:
            errors.append(repr(exc))
        try:
            await runner.cleanup()
        except BaseException as exc:
            errors.append(repr(exc))
        report["cleanup_errors"] = errors
        if errors:
            report["status"] = "failed"
        (out / "qualification.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--header-evidence", type=Path, required=True)
    parser.add_argument("--creator-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-port", type=int, default=28901)
    asyncio.run(main(parser.parse_args()))
