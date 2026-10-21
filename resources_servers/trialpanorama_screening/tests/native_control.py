"""Actual native CLI with saved response fixtures; no external inference or performance claim."""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import psutil
import yaml
from aiohttp import web


async def main(out, saved, creator):
    out = out.absolute()
    out.mkdir(exist_ok=False)
    source = Path.cwd()
    rows = [json.loads(s) for s in saved.read_text().splitlines()][:2]
    samples = [
        {
            "task_id": str(r["verifier_metadata"]["id"]),
            "responses_create_params": r["responses_create_params"],
            "verifier_metadata": r["verifier_metadata"],
        }
        for r in rows
    ]
    sample = out / "sample.jsonl"
    sample.write_text("".join(json.dumps(r) + "\n" for r in samples))
    calls = []

    async def provider(request):
        body = await request.json()
        index = next(i for i, r in enumerate(rows) if r["responses_create_params"]["input"] == body["input"])
        assert index not in [c["index"] for c in calls]
        assert "verifier_metadata" not in body
        calls.append({"index": index, "request": body})
        (out / "provider-calls.json").write_text(json.dumps(calls, indent=2))
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
        "resources_servers/trialpanorama_screening",
        "environment_servers/legacy_agent",
    ]:
        p = out / "venvs" / component / ".venv"
        p.parent.mkdir(parents=True)
        p.symlink_to(Path(sys.executable).absolute().parent.parent)
    config = {
        "config_paths": ["benchmarks/trialpanorama_screening/config.yaml"],
        "trialpanorama_screening": {
            "resources_servers": {
                "trialpanorama_screening": {
                    "datasets": [
                        {
                            "name": "trialpanorama_screening",
                            "type": "benchmark",
                            "jsonl_fpath": str(sample),
                            "prepare_script": "benchmarks/trialpanorama_screening/prepare.py",
                            "num_repeats": 1,
                        }
                    ]
                }
            }
        },
        "policy_model_name": rows[0]["response"]["model"],
        "policy_api_key": "loopback-only",
        "policy_base_url": f"http://127.0.0.1:{port}/v1",
        "provider_dispatch_guard": True,
        "observability_enabled": True,
        "model_call_capture_dir": str(out / "capture"),
        "head_server": {"host": "127.0.0.1", "port": 25300},
        "port_range_low": 25301,
        "port_range_high": 25330,
        "uv_venv_dir": str(out / "venvs"),
        "skip_venv_if_present": True,
        "use_cached_prepared_benchmarks": True,
        "dispatch_budget_s": 90,
        "drain_margin_s": 60,
        "policy_model": {
            "responses_api_models": {
                "openai_model": {
                    "max_concurrent_requests": 1,
                    "upstream_request_timeout_seconds": 30,
                    "upstream_max_num_tries": 1,
                }
            }
        },
    }
    (out / "runtime.yaml").write_text(yaml.safe_dump(config))
    env = dict(
        os.environ,
        PYTHONPATH=str(source),
        NEMO_GYM_EXTRA_ROOTS=str(source),
        TRIALPANORAMA_SOURCE=str(creator.absolute()),
        CAMPAIGN_SAMPLE_JSONL=str(sample),
        CAMPAIGN_SAMPLE_SHA256=hashlib.sha256(sample.read_bytes()).hexdigest(),
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
        "2",
        "--temperature",
        "0.1",
        "--max-output-tokens",
        "16384",
        "--concurrency",
        "1",
        "--output",
        str(out / "results.jsonl"),
    ]
    (out / "command.json").write_text(json.dumps(command))
    child = None
    paths = [
        "resources_servers/trialpanorama_screening/app.py",
        "resources_servers/trialpanorama_screening/grading.py",
        "resources_servers/trialpanorama_screening/configs/trialpanorama_screening.yaml",
        "responses_api_agents/simple_agent/app.py",
        "responses_api_models/openai_model/app.py",
    ]
    (out / "source.json").write_text(
        json.dumps(
            {
                "files": {n: hashlib.sha256((source / n).read_bytes()).hexdigest() for n in paths},
                "saved_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
                "creator": str(creator),
            },
            indent=2,
        )
    )
    try:
        with (out / "run.log").open("w") as f:
            child = await asyncio.create_subprocess_exec(*command, env=env, stdout=f, stderr=f)
            code = await asyncio.wait_for(child.wait(), 150)
        (out / "exit.json").write_text(json.dumps({"exit": code}))
        assert code == 0
        results = [json.loads(s) for s in (out / "results.jsonl").read_text().splitlines()]
        captures = [
            json.loads(s) for p in (out / "capture").glob("*.capture.jsonl") for s in p.read_text().splitlines()
        ]
        assert len(results) == len(calls) == len(captures) == 2
        assert all(c["status_code"] == 200 and len(c["provider_attempts"]) == 1 for c in captures)
        assert not list((out / "capture").glob("*pending*"))
        from resources_servers.trialpanorama_screening.grading import grade

        for actual, row in zip(results, rows):
            expected = grade(
                row["response"]["output"][0]["content"][0]["text"], creator_root=creator, **row["verifier_metadata"]
            )
            for key in ["reward", "mask_sample", "diagnostics", "metadata"]:
                assert actual[key] == expected[key], key
        (out / "acceptance.json").write_text(
            json.dumps(
                {
                    "results": 2,
                    "physical_calls": 2,
                    "external_inference": 0,
                    "scope": "actual native CLI and captured saved fixtures; not model performance",
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
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--saved", type=Path, required=True)
    p.add_argument("--creator", type=Path, required=True)
    a = p.parse_args()
    asyncio.run(main(a.output, a.saved, a.creator))
