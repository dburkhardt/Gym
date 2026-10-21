# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual native CLI checkpoint control with unchanged saved QCBench responses."""

import argparse
import asyncio
import json
import os
import signal
import sys
from pathlib import Path

import psutil
import yaml
from aiohttp import web


async def main(out, saved, creator):
    out = out.resolve()
    out.mkdir(exist_ok=False)
    source = Path.cwd()
    original = [json.loads(line) for line in saved.read_text().splitlines()]
    row = original[0]
    sample = {key: row[key] for key in ("uuid", "responses_create_params", "verifier_metadata")}
    sample["task_id"] = row["uuid"]
    (out / "sample.jsonl").write_text(json.dumps(sample) + "\n")
    calls = []
    child = None
    owned = []
    sentinel = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(300)")

    async def provider(request):
        payload = await request.json()
        calls.append(payload)
        (out / "provider-calls.json").write_text(json.dumps(calls))
        if len(calls) > 1:
            raise AssertionError("Unexpected regeneration after checkpoint")
        return web.json_response(row["response"])

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
        "resources_servers/campaign_creator_scoring",
        "environment_servers/legacy_agent",
    ]:
        path = out / "venvs" / component / ".venv"
        path.parent.mkdir(parents=True)
        path.symlink_to(Path(sys.executable).absolute().parent.parent)
    config = {
        "config_paths": ["benchmarks/qcbench/config.yaml"],
        "qcbench_resources": {
            "resources_servers": {
                "campaign_creator_scoring": {
                    "upstream_repo": str(creator.resolve()),
                    "datasets": [
                        {
                            "name": "qcbench",
                            "type": "benchmark",
                            "jsonl_fpath": str(out / "sample.jsonl"),
                            "prepare_script": "benchmarks/qcbench/prepare.py",
                            "num_repeats": 1,
                        }
                    ],
                }
            }
        },
        "policy_model_name": row["response"]["model"],
        "policy_api_key": "local-control",
        "policy_base_url": f"http://127.0.0.1:{port}/v1",
        "provider_dispatch_guard": True,
        "observability_enabled": True,
        "model_call_capture_dir": str(out / "capture"),
        "head_server": {"host": "127.0.0.1", "port": 32100},
        "port_range_low": 32101,
        "port_range_high": 32120,
        "uv_venv_dir": str(out / "venvs"),
        "skip_venv_if_present": True,
        "use_cached_prepared_benchmarks": True,
        "checkpoint": {"enabled": True, "control_auth_token": "local-checkpoint-control"},
        "policy_model": {
            "responses_api_models": {
                "openai_model": {
                    "checkpoint_policy": True,
                    "upstream_max_num_tries": 1,
                    "upstream_request_timeout_seconds": 600,
                }
            }
        },
        "route_failures_to_sidecar": True,
        "require_complete": False,
    }
    (out / "runtime.yaml").write_text(yaml.safe_dump(config))
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
        row["response"]["model"],
        "--split",
        "benchmark",
        "--limit",
        "1",
        "--concurrency",
        "1",
        "--health-check-workers",
        "1",
        "--output",
        str(out / "results.jsonl"),
    ]
    hooks = out / "timing-hook"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text("""import asyncio, os
from pathlib import Path
from nemo_gym._checkpoint.agent import LegacyRun
original = LegacyRun.boundary
async def paused(self, state):
    marker = os.environ.get('GYM_TEST_BOUNDARY_MARKER')
    if marker and state.get('next') == 'verify':
        Path(marker).write_text('completed solver; before verifier')
        while not self._steps.closed:
            await asyncio.sleep(0.01)
    await original(self, state)
LegacyRun.boundary = paused
""")
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join([str(hooks), str(source)]),
        GYM_TEST_BOUNDARY_MARKER=str(out / "boundary-ready"),
        NEMO_GYM_EXTRA_ROOTS=str(source),
        CAMPAIGN_UPSTREAM_REPO=str(creator.resolve()),
    )
    (out / "command.json").write_text(json.dumps(command))
    try:
        with (out / "initial.log").open("w") as log:
            child = await asyncio.create_subprocess_exec(
                *(command + ["--checkpoint-dir", str(out / "checkpoint")]),
                env=env,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
            )
            (out / "process.json").write_text(json.dumps({"pid": child.pid}))
            for _ in range(18000):
                if child.returncode is not None:
                    raise RuntimeError("Initial CLI exited before checkpoint; inspect initial.log")
                owned = psutil.Process(child.pid).children(recursive=True)
                if (out / "boundary-ready").exists():
                    child.send_signal(signal.SIGUSR1)
                    break
                await asyncio.sleep(0.01)
            else:
                raise TimeoutError("Native capture did not appear")
            for _ in range(6000):
                if (out / "checkpoint/evaluation.json").exists():
                    break
                if child.returncode is not None:
                    raise RuntimeError("Checkpoint not published; inspect initial.log")
                await asyncio.sleep(0.01)
            else:
                raise TimeoutError("Checkpoint publication timeout")
            for process in reversed(owned):
                if process.is_running():
                    process.kill()
            child.kill()
            await child.wait()
            _, alive = await asyncio.to_thread(psutil.wait_procs, owned, timeout=10)
            assert not [p for p in alive if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
            assert sentinel.returncode is None
        env.pop("GYM_TEST_BOUNDARY_MARKER", None)
        with (out / "restore.log").open("w") as log:
            child = await asyncio.create_subprocess_exec(
                *(command + ["--resume", "--restore-checkpoint", str(out / "checkpoint")]),
                env=env,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
            )
            assert await asyncio.wait_for(child.wait(), 120) == 0
        result = json.loads((out / "results.jsonl").read_text().splitlines()[0])
        assert result["reward"] == row["reward"] and len(calls) == 1
        for key in ("scoring_details", "mask_sample", "failure_kind"):
            assert result.get(key) == row.get(key), key
        from nemo_gym.openai_utils import NeMoGymResponse

        assert NeMoGymResponse.model_validate(result["response"]) == NeMoGymResponse.model_validate(row["response"])
        assert sentinel.returncode is None
        (out / "acceptance.json").write_text(
            json.dumps({"calls": len(calls), "reward": result["reward"], "external_inference": 0, "status": "passed"})
        )
    finally:
        if child and child.returncode is None:
            owned = psutil.Process(child.pid).children(recursive=True)
            for process in reversed(owned):
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            child.kill()
            await child.wait()
        sentinel.terminate()
        await sentinel.wait()
        await runner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--saved", type=Path, required=True)
    parser.add_argument("--creator", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.output, args.saved, args.creator))
