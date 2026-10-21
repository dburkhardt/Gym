# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native CLI sidecar/cooldown control with saved SuperChem fixture responses."""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import psutil
import yaml
from aiohttp import web


async def main(out: Path, saved: Path) -> None:
    out = out.absolute()
    out.mkdir(exist_ok=False)
    source = Path.cwd()
    identity_paths = [
        "tests/e2e/coordinator/rate_limit_peer_control.py",
        "configs/eval/rate_limit_peer_safe.yaml",
        "nemo_gym/rollout_collection.py",
        "responses_api_models/openai_model/app.py",
        "resources_servers/superchem/app.py",
    ]
    (out / "source.json").write_text(
        json.dumps(
            {
                "source_root": str(source),
                "interpreter": sys.executable,
                "saved_fixture_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
                "files": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in identity_paths},
            },
            indent=2,
        )
    )
    original = [json.loads(line) for line in saved.read_text().splitlines()]
    budget = next(row for row in original if row.get("failure_kind") == "model_budget")
    healthy = next(row for row in original if row.get("failure_kind") is None)
    rejected = next(row for row in original if row["task_id"] not in {budget["task_id"], healthy["task_id"]})
    rows = [rejected, budget, healthy]
    samples = [{key: row[key] for key in ["task_id", "responses_create_params", "verifier_metadata"]} for row in rows]
    (out / "sample.jsonl").write_text("".join(json.dumps(row) + "\n" for row in samples))
    accepted = asyncio.Event()
    calls = []
    peer_finished = False
    child = None

    async def provider(request):
        nonlocal peer_finished
        body = await request.json()
        index = next(i for i, row in enumerate(rows) if row["responses_create_params"]["input"] == body["input"])
        assert index not in [call["index"] for call in calls], "Unexpected resend"
        calls.append({"index": index, "time": time.monotonic(), "request": body})
        (out / "provider-calls.json").write_text(json.dumps(calls))
        if index == 0:
            await asyncio.wait_for(accepted.wait(), 10)
            return web.json_response(
                {"error": {"message": "fixture quota", "type": "rate_limit"}}, status=429, headers={"Retry-After": "1"}
            )
        if index == 1:
            accepted.set()
            await asyncio.sleep(0.1)
            peer_finished = True
        return web.json_response(rows[index]["response"])

    app = web.Application()
    app.router.add_post("/v1/responses", provider)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    for component in [
        "responses_api_models/openai_model",
        "responses_api_agents/simple_agent",
        "resources_servers/superchem",
        "environment_servers/legacy_agent",
    ]:
        p = out / "venvs" / component / ".venv"
        p.parent.mkdir(parents=True)
        p.symlink_to(Path(sys.executable).absolute().parent.parent)
    config = {
        "config_paths": ["benchmarks/superchem/config.yaml", "configs/eval/rate_limit_peer_safe.yaml"],
        "superchem": {
            "resources_servers": {
                "superchem": {
                    "datasets": [
                        {
                            "name": "superchem",
                            "type": "benchmark",
                            "jsonl_fpath": str(out / "sample.jsonl"),
                            "prepare_script": "benchmarks/superchem/prepare.py",
                            "num_repeats": 1,
                        }
                    ]
                }
            }
        },
        "policy_model_name": rows[0]["response"]["model"],
        "policy_api_key": "local-fixture",
        "policy_base_url": f"http://127.0.0.1:{port}/v1",
        "provider_dispatch_guard": True,
        "observability_enabled": True,
        "model_call_capture_dir": str(out / "capture"),
        "head_server": {"host": "127.0.0.1", "port": 32700},
        "port_range_low": 32701,
        "port_range_high": 32730,
        "uv_venv_dir": str(out / "venvs"),
        "skip_venv_if_present": True,
        "use_cached_prepared_benchmarks": True,
        "dispatch_budget_s": 90,
        "drain_margin_s": 60,
        "policy_model": {
            "responses_api_models": {
                "openai_model": {"max_concurrent_requests": 2, "upstream_request_timeout_seconds": 60}
            }
        },
    }
    (out / "runtime.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(source),
        NEMO_GYM_EXTRA_ROOTS=str(source),
        CAMPAIGN_SAMPLE_JSONL=str(out / "sample.jsonl"),
        CAMPAIGN_SAMPLE_SHA256=hashlib.sha256((out / "sample.jsonl").read_bytes()).hexdigest(),
        GYM_PROVIDER_COOLDOWN_DIRECTORY=str(out / "cooldown"),
        GYM_PROVIDER_COOLDOWN_NAMESPACE="local-fixture",
    )
    command = [
        sys.executable,
        "-c",
        "from nemo_gym.cli.main import main; main()",
        "eval",
        "run",
        "--model-type",
        "openai_model",
        "--config",
        str(out / "runtime.yaml"),
        "--model",
        config["policy_model_name"],
        "--split",
        "benchmark",
        "--limit",
        "3",
        "--temperature",
        "0.1",
        "--max-output-tokens",
        "16384",
        "--concurrency",
        "2",
        "--output",
        str(out / "results.jsonl"),
    ]
    (out / "command.json").write_text(json.dumps(command))
    try:
        with (out / "run.log").open("w") as log:
            child = await asyncio.create_subprocess_exec(
                *command, env=env, stdout=log, stderr=asyncio.subprocess.STDOUT
            )
            code = await asyncio.wait_for(child.wait(), 120)
        (out / "exit.json").write_text(json.dumps({"code": code}))
        assert code == 0, f"Native CLI exited {code}; preserve run.log"
        results = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
        failures = [json.loads(line) for line in (out / "results_failures.jsonl").read_text().splitlines()]
        captures = [
            json.loads(line) for p in (out / "capture").glob("*.capture.jsonl") for line in p.read_text().splitlines()
        ]
        assert len(results) == 2 and len(failures) == 1 and len(captures) == 3
        assert "reward" not in failures[0] and failures[0]["_ng_task_index"] == 0
        assert peer_finished and sorted(call["index"] for call in calls) == [0, 1, 2]
        times = {call["index"]: call["time"] for call in calls}
        assert times[2] - times[0] >= 0.9
        assert not list((out / "capture").glob("*pending*"))
        assert sorted(c["status_code"] for c in captures) == [200, 200, 429]
        assert all(len(c["provider_attempts"]) == 1 for c in captures)
        actual = {row["task_id"]: row for row in results}
        for row in [budget, healthy]:
            for field in ["reward", "mask_sample", "failure_kind", "diagnostics"]:
                assert actual[row["task_id"]].get(field) == row.get(field)
        pids = set(
            map(int, re.findall(r"(?:Started|Finished) server process \[(\d+)\]", (out / "run.log").read_text()))
        )
        alive = [pid for pid in pids if psutil.pid_exists(pid)]
        assert not alive
        (out / "acceptance.json").write_text(
            json.dumps(
                {
                    "exit": code,
                    "results": 2,
                    "infrastructure_failures": 1,
                    "physical_calls": 3,
                    "peer_completed": True,
                    "cooldown_wait_seconds": times[2] - times[0],
                    "pending": 0,
                    "owned_pids_absent": sorted(pids),
                    "external_inference": 0,
                    "scope": "Actual native CLI/model/capture/scorer; provider responses are saved fixtures, not new scientific results",
                },
                indent=2,
            )
        )
    finally:
        if child is not None and child.returncode is None:
            owned = psutil.Process(child.pid).children(recursive=True)
            for process in reversed(owned):
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            child.kill()
            await child.wait()
        await runner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--saved", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.output, args.saved))
