# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import psutil
import yaml
from aiohttp import web


async def main(out, *, logged=False):
    out = out.absolute()
    out.mkdir(exist_ok=False)
    source = Path.cwd()
    calls = []
    children = []
    runner = None
    sentinel = None
    report = {"status": "failed", "external_inference": 0}

    async def provider(req):
        body = await req.json()
        calls.append(body)
        (out / "provider-calls.json").write_text(json.dumps(calls))
        if len(calls) > 1:
            await asyncio.sleep(180)
        return web.json_response(
            {
                "id": "fixture1",
                "object": "response",
                "created_at": 0,
                "model": "fixture",
                "status": "completed",
                "output": [
                    {
                        "id": "m1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Option 1", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )

    async def stop_tree(child):
        known = psutil.Process(child.pid).children(recursive=True) if psutil.pid_exists(child.pid) else []
        for p in known:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        if child.returncode is None:
            child.kill()
        await child.wait()
        await asyncio.sleep(1)
        return [{"pid": p.pid, "alive": p.is_running() and p.status() != psutil.STATUS_ZOMBIE} for p in known]

    try:
        sentinel = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(180)", start_new_session=True
        )
        app = web.Application()
        app.router.add_post("/v1/responses", provider)
        runner = web.AppRunner(app, shutdown_timeout=0.1)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        for c in [
            "responses_api_models/openai_model",
            "responses_api_agents/simple_agent",
            "resources_servers/proteinlmbench",
            "environment_servers/legacy_agent",
        ]:
            p = out / "venvs" / c / ".venv"
            p.parent.mkdir(parents=True)
            p.symlink_to(Path(sys.executable).absolute().parent.parent)
        rows = [
            {
                "task_id": f"fixture-{i}",
                "responses_create_params": {
                    "input": [{"role": "user", "content": f"Benign fixture {i}: choose Option 1."}]
                },
                "verifier_metadata": {"expected_answer": 1, "option_count": 2},
            }
            for i in range(3)
        ]
        (out / "sample.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        config = {
            "config_paths": ["benchmarks/proteinlmbench/config.yaml"],
            "proteinlmbench": {
                "resources_servers": {
                    "proteinlmbench": {
                        "datasets": [
                            {
                                "name": "proteinlmbench",
                                "type": "benchmark",
                                "jsonl_fpath": str(out / "sample.jsonl"),
                                "prepare_script": "benchmarks/proteinlmbench/prepare.py",
                                "num_repeats": 1,
                            }
                        ]
                    }
                }
            },
            "policy_model_name": "fixture",
            "policy_api_key": "fixture",
            "policy_base_url": f"http://127.0.0.1:{port}/v1",
            "provider_dispatch_guard": True,
            "observability_enabled": True,
            "model_call_capture_dir": str(out / "capture"),
            "head_server": {"host": "127.0.0.1", "port": 28800},
            "port_range_low": 28801,
            "port_range_high": 28820,
            "uv_venv_dir": str(out / "venvs"),
            "skip_venv_if_present": True,
            "use_cached_prepared_benchmarks": True,
            "policy_model": {
                "responses_api_models": {
                    "openai_model": {"upstream_max_num_tries": 1, "upstream_request_timeout_seconds": 120}
                }
            },
        }
        if logged:
            config["nemo_gym_log_dir"] = str(out / "server-logs")
        (out / "runtime.yaml").write_text(yaml.safe_dump(config))
        env = os.environ.copy()
        env.update(PYTHONPATH=str(source), NEMO_GYM_EXTRA_ROOTS=str(source))
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
            "fixture",
            "--split",
            "benchmark",
            "--limit",
            "3",
            "--num-repeats",
            "1",
            "--concurrency",
            "1",
            "--health-check-workers",
            "1",
            "--output",
            str(out / "results.jsonl"),
        ]
        (out / "command.json").write_text(json.dumps(command))
        log = (out / "initial.log").open("w")
        child = await asyncio.create_subprocess_exec(*command, env=env, stdout=log, stderr=asyncio.subprocess.STDOUT)
        children.append(child)
        for _ in range(1800):
            if len(calls) == 2 and (out / "results.jsonl").exists() and (out / "results.jsonl").read_text().strip():
                break
            if child.returncode is not None:
                raise RuntimeError("initial native process terminated early")
            await asyncio.sleep(0.1)
        else:
            raise TimeoutError("did not reach two calls and saved result")
        prior = (out / "results.jsonl").read_bytes()
        (out / "before-results.jsonl").write_bytes(prior)
        shutil.copytree(out / "capture", out / "before-capture")
        descendants = psutil.Process(child.pid).children(recursive=True)
        report["killed_coordinator_pid"] = child.pid
        report["owned_descendants"] = [p.pid for p in descendants]
        child.kill()
        await child.wait()
        report["coordinator_exit"] = child.returncode
        # Settle exact orphaned children after killing coordinator; do not reuse their memory.
        for p in descendants:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        await asyncio.sleep(2)
        report["orphan_cleanup_scope"] = (
            "exact captured descendants killed after top-level SIGKILL; not autonomous lifecycle proof"
        )
        log2 = (out / "resume.log").open("w")
        resume = await asyncio.create_subprocess_exec(
            *(command + ["--resume"]), env=env, stdout=log2, stderr=asyncio.subprocess.STDOUT
        )
        children.append(resume)
        resume_children = {}
        for _ in range(900):
            if psutil.pid_exists(resume.pid):
                for owned in psutil.Process(resume.pid).children(recursive=True):
                    resume_children[(owned.pid, owned.create_time())] = owned
            if resume.returncode is not None or len(calls) > 2:
                break
            await asyncio.sleep(0.1)
        report["resume_owned_children_alive"] = [
            p.pid for p in resume_children.values() if p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        ]
        report["capture_evidence_preserved"] = all(
            (out / "capture" / p.name).read_bytes() == p.read_bytes()
            for p in (out / "before-capture").iterdir()
            if p.is_file()
        )
        report["source_hashes"] = {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ["nemo_gym/base_responses_api_model.py", "nemo_gym/rollout_collection.py"]
        }
        report["untouched_task_ids"] = ["fixture-2"]
        report["unrelated_sentinel_survived"] = sentinel.returncode is None
        report["logged_server_launch"] = logged
        report["resume_exit"] = resume.returncode
        report["calls_after_resume"] = len(calls)
        report["completed_result_preserved"] = (out / "results.jsonl").read_bytes() == prior
        report["resumed_unknown_resend"] = len(calls) > 2
        report["status"] = (
            "passed"
            if len(calls) == 2
            and resume.returncode not in (None, 0)
            and report["completed_result_preserved"]
            and report["capture_evidence_preserved"]
            and not report["resume_owned_children_alive"]
            and report["unrelated_sentinel_survived"]
            else "failed"
        )
    finally:
        report["cleanup"] = []
        for child in children:
            if child.returncode is None:
                report["cleanup"].extend(await stop_tree(child))
        if sentinel is not None and sentinel.returncode is None:
            sentinel.terminate()
            await sentinel.wait()
        if runner:
            await runner.cleanup()
        (out / "receipt.json").write_text(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise RuntimeError(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    asyncio.run(main(p.parse_args().output))
