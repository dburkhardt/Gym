# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixed saved-response recovery through native CLI and real SciCode containers.

Run under the maintained process supervisor with an 1800-second combined bound.
Assets are explicit paths; no credentials, package installation or image pulls.
The 30-second loopback transport timeout is control-specific, not a live budget.
"""

import argparse
import asyncio
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

import yaml
from aiohttp import web


MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"
IMAGE = "sha256:2fae326d6d1d11e55b828f200bfcd349bf392e070fc69e7721a87d152db97918"


def save(name, x):
    (R / name).write_text(json.dumps(x, indent=2))


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fixture_reply(ordinal, body, molrequest, molresponse, captures):
    """Exact four-call transcript; no fallback or replay beyond the declared attempt."""
    if not 1 <= ordinal <= 4:
        raise ValueError("Unexpected extra fixture call")
    index = 0 if ordinal in (2, 3) else 1
    expected = molrequest if ordinal == 1 else dict(captures[index]["request"], model=MODEL)
    if body != expected:
        raise ValueError("Literal fixture request mismatch")
    if ordinal == 2:
        return 429, {"error": {"message": "controlled definitive rejection"}}
    return 200, molresponse if ordinal == 1 else captures[index]["response"]


def require_container_absence(name, probe):
    if probe.returncode == 0 or "no such" not in probe.stderr.lower() or name not in probe.stderr:
        raise RuntimeError("Scientific container absence not established")


def read_planned_inputs(output, moljson_uuid):
    """Read eval run --plan-only's artifact, not execution materialized inputs."""
    path = output.with_stem(output.stem + "_planned_inputs").with_suffix(".jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 2 or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Expected exactly two planned task objects")
    if [row.get("task_source") for row in rows] != ["moljson_resources", "scicode_benchmark_agent"]:
        raise ValueError("Unexpected planned route order")
    if rows[0].get("uuid") != moljson_uuid or rows[0].get("task_id") != moljson_uuid or rows[1].get("task_id") != "20":
        raise ValueError("Unexpected planned task identity")
    if [row.get("_ng_task_index") for row in rows] != [0, 1]:
        raise ValueError("Unexpected planned task indices")
    return rows


def remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("Combined control deadline expired before next native invocation")
    return value


async def main():
    h5 = args.h5.resolve()
    assert sha(h5) == "48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890"
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", IMAGE], timeout=15))[0]
    assert image["Id"] == IMAGE
    assets = {
        name: sha(getattr(args, name))
        for name in ["scicode_input", "scicode_captures", "moljson_input", "moljson_result", "moljson_request", "h5"]
    }
    save("assets.json", assets)
    save(
        "budget.json",
        dict(
            outer_seconds=1800,
            work_seconds=1740,
            final_settlement_seconds=60,
            fixture_calls=4,
            fixture_transport_timeout=30,
            scientific_substeps=2,
            scientific_timeout_per_substep=30,
            declared_provider_and_scientific_seconds=180,
            operational_remainder_seconds=1620,
            scope="Operational abort, not live-provider capacity; all invocations share one deadline",
        ),
    )
    captures = [json.loads(line) for line in args.scicode_captures.read_text().splitlines()]
    assert len(captures) == 2
    row = json.loads(args.scicode_input.read_text())
    row.pop("_ng_rollout_id", None)
    row["task_id"] = "20"
    (R / "sample.jsonl").write_text(json.dumps(row) + "\n")
    molrow = [json.loads(line) for line in args.moljson_input.read_text().splitlines()]
    assert len(molrow) == 1
    (R / "moljson.jsonl").write_text(json.dumps(molrow[0]) + "\n")
    molresult = [json.loads(line) for line in args.moljson_result.read_text().splitlines()]
    assert len(molresult) == 1 and molresult[0]["reward"] == 1
    molrequest = json.loads(args.moljson_request.read_text())
    if isinstance(molrequest, list):
        molrequest = molrequest[0]
    from nemo_gym.openai_utils import NeMoGymResponse

    for response in [molresult[0]["response"], *[capture["response"] for capture in captures]]:
        NeMoGymResponse.model_validate(response)
    calls = []
    failure = []

    async def provider(req):
        body = await req.json()
        calls.append(body)
        save("provider-calls.json", calls)
        try:
            status, payload = fixture_reply(len(calls), body, molrequest, molresult[0]["response"], captures)
        except ValueError as error:
            failure.append(str(error))
            save("mismatch.json", {"ordinal": len(calls), "actual": body, "error": str(error)})
            return web.json_response({"error": str(error)}, status=500)
        return web.json_response(payload, status=status)

    app = web.Application()
    app.router.add_post("/v1/responses", provider)
    runner = web.AppRunner(app, shutdown_timeout=1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(S), NEMO_GYM_EXTRA_ROOTS=str(S), NEMO_GYM_SERVICE_OWNERSHIP_MODE="systemd")
    for component in [
        "responses_api_models/openai_model",
        "responses_api_agents/scicode_agent",
        "resources_servers/scicode",
        "environment_servers/legacy_agent",
        "responses_api_agents/simple_agent",
        "resources_servers/campaign_creator_scoring",
    ]:
        p = R / "venvs" / component / ".venv"
        p.parent.mkdir(parents=True)
        p.symlink_to(pathlib.Path(sys.executable).absolute().parent.parent)
    from nemo_gym.scicode_restart import inspect_component_runtime

    imports = {}
    for component in [
        "responses_api_models/openai_model",
        "responses_api_agents/scicode_agent",
        "resources_servers/scicode",
        "environment_servers/legacy_agent",
        "responses_api_agents/simple_agent",
        "resources_servers/campaign_creator_scoring",
    ]:
        imports[component] = await asyncio.to_thread(
            inspect_component_runtime, S, S / component, pathlib.Path(sys.executable)
        )
    save("actual-imports.json", imports)
    config = {
        "config_paths": ["benchmarks/moljson/config.yaml", "benchmarks/scicode/config.yaml"],
        "policy_model_name": MODEL,
        "policy_api_key": "local-fixture-only",
        "policy_base_url": f"http://127.0.0.1:{port}/v1",
        "provider_dispatch_guard": True,
        "observability_enabled": True,
        "model_call_capture_dir": str(R / "capture"),
        "head_server": {"host": "127.0.0.1", "port": args.port},
        "port_range_low": args.port + 1,
        "port_range_high": args.port + 30,
        "uv_venv_dir": str(R / "venvs"),
        "skip_venv_if_present": True,
        "use_cached_prepared_benchmarks": True,
        "route_failures_to_sidecar": True,
        "require_complete": False,
        "scicode_benchmark_agent": {
            "responses_api_agents": {
                "scicode_agent": {
                    "datasets": [
                        {
                            "name": "scicode",
                            "type": "benchmark",
                            "jsonl_fpath": str(R / "sample.jsonl"),
                            "prepare_script": "benchmarks/scicode/prepare.py",
                            "prompt_config": None,
                            "num_repeats": 1,
                        }
                    ]
                }
            }
        },
        "scicode_benchmark_resources_server": {
            "resources_servers": {
                "scicode": {
                    "sandbox_image": IMAGE,
                    "num_processes": 1,
                    "sandbox_cpu": 1.0,
                    "sandbox_memory_mib": 4096,
                    "timeout_secs": 30.0,
                    "test_data_fpath": str(h5),
                    "execution_artifact_dir": str(R / "scientific-execution"),
                }
            }
        },
        "policy_model": {
            "responses_api_models": {
                "openai_model": {
                    "upstream_max_num_tries": 1,
                    "upstream_request_timeout_seconds": 30,
                    "upstream_retry_policy": {"max_attempts": 1},
                }
            }
        },
    }
    config["moljson_resources"] = {
        "resources_servers": {
            "campaign_creator_scoring": {
                "upstream_repo": str(args.moljson_repo.resolve()),
                "datasets": [
                    {
                        "name": "moljson",
                        "type": "benchmark",
                        "jsonl_fpath": str(R / "moljson.jsonl"),
                        "prepare_script": "benchmarks/moljson/prepare.py",
                        "num_repeats": 1,
                    }
                ],
            }
        }
    }
    command = [
        sys.executable,
        "-c",
        "from nemo_gym.cli.main import main; main()",
        "eval",
        "run",
        "--config",
        str(R / "runtime.yaml"),
        "--model-type",
        "openai_model",
        "--split",
        "benchmark",
        "--output",
        str(R / "results.jsonl"),
        "--num-repeats",
        "1",
        "--concurrency",
        "1",
        "--health-check-workers",
        "1",
    ]
    save("command.json", command)

    async def run(label, extra, expected=0):
        remaining(DEADLINE)
        (R / "runtime.yaml").write_text(yaml.safe_dump(config))
        with (R / (label + ".log")).open("w") as f:
            child = await asyncio.create_subprocess_exec(
                *command, *extra, cwd=S, env=env, stdout=f, stderr=asyncio.subprocess.STDOUT
            )
            save(label + "-pid.json", {"pid": child.pid})
            try:
                code = await asyncio.wait_for(child.wait(), remaining(DEADLINE))
            finally:
                if child.returncode is None:
                    child.send_signal(signal.SIGINT)
                    try:
                        await asyncio.wait_for(child.wait(), 60)
                    except asyncio.TimeoutError:
                        child.kill()
                        await child.wait()
            save(label + "-exit.json", {"returncode": code})
            if code != expected:
                raise RuntimeError(label + " native CLI unexpected exit " + str(code))

    try:
        await run("plan", ["--plan-only"])
        assert not calls
        read_planned_inputs(R / "results.jsonl", molrow[0]["uuid"])

        # rollout_collection's zero-results guard raises only when no rollout succeeded.
        # The accepted pure SciCode control exits 1 there; this mixed prefix makes exit 0 correct.
        await run("initial", [])
        assert len(calls) == 2 and not failure
        from nemo_gym.batch_restart import digest

        status = json.loads((R / "results/batch_status.json").read_text())
        assert status["execution_binding"]["restart_scope_qualified"]
        failed_path = R / "results_failures.jsonl"
        failed = [json.loads(line) for line in failed_path.read_text().splitlines()]
        assert len(failed) == 1
        prefix = (R / "results.jsonl").read_bytes()
        assert len(prefix.splitlines()) == 1
        first_result = json.loads(prefix)
        assert (
            first_result["uuid"] == molrow[0]["uuid"]
            and first_result["reward"] == 1
            and not first_result.get("mask_sample", False)
        )
        old_failure = failed_path.read_bytes()
        original = {p.name: p.read_bytes() for p in (R / "capture").iterdir() if p.is_file()}
        save(
            "restart-selection.json",
            {
                "execution_binding_sha256": digest(status["execution_binding"]),
                "failures": [
                    {
                        "capture_key": f"{failed[0]['_ng_task_index']}-{failed[0]['_ng_rollout_index']}",
                        "failure_sha256": digest(failed[0]),
                    }
                ],
            },
        )
        config["guarded_restart_receipt_fpath"] = str(R / "restart-selection.json")
        await run("restart", ["--resume"])
        assert len(calls) == 4 and not failure
        result = [json.loads(line) for line in (R / "results.jsonl").read_text().splitlines()]
        assert len(result) == 2 and result[1]["_ng_attempt_index"] == 1
        assert (R / "results.jsonl").read_bytes().startswith(prefix)
        assert (
            result[1]["reward"] == 1
            and result[1]["step_results"] == [True, True]
            and not result[1].get("mask_sample", False)
        )
        assert failed_path.read_bytes() == old_failure
        assert all((R / "capture" / name).read_bytes() == data for name, data in original.items())
        physical = [
            json.loads(line)
            for path in (R / "capture").glob("*.capture.jsonl")
            for line in path.read_text().splitlines()
        ]
        assert len(physical) == 4 and sorted(c["status_code"] for c in physical) == [200, 200, 200, 429]
        assert all(len(c["provider_attempts"]) == 1 for c in physical)
        assert not list((R / "capture").glob("*pending*"))
        completed = (R / "results.jsonl").read_bytes()
        config.pop("guarded_restart_receipt_fpath")
        await run("completed-resume", ["--resume"])
        assert len(calls) == 4 and (R / "results.jsonl").read_bytes() == completed
        receipts = [json.loads(p.read_text()) for p in (R / "scientific-execution").glob("*/lifecycle.json")]
        assert len(receipts) == 2 and all(p["cleanup_confirmed"] for p in receipts)
        cleanup = {"containers": [], "services": []}
        for receipt in receipts:
            name = receipt["sandbox"]["sandbox_id"]
            probe = subprocess.run(
                ["docker", "inspect", name], capture_output=True, text=True, timeout=min(15, remaining(DEADLINE))
            )
            require_container_absence(name, probe)
            cleanup["containers"].append(name)
        from nemo_gym.cli.service_ownership import _scope_identity

        ownership = list((R / "results/native_services").glob("*.json"))
        assert ownership, "Missing native ownership receipts"
        for path in ownership:
            receipt = json.loads(path.read_text())
            assert receipt["cleanup_confirmed"] and receipt.get("mode") == "systemd"
            assert _scope_identity(receipt) is None, "Owned service scope still present"
            cleanup["services"].append({"file": path.name, "sha256": sha(path), "systemd": receipt["systemd"]})
        save("cleanup.json", cleanup)
        save(
            "acceptance.json",
            dict(
                status="passed",
                external_inference=0,
                fixture_physical_calls=len(calls),
                new_attempt=1,
                reward=1,
                step_results=[True, True],
                completed_resume_new_calls=0,
                old_failure_preserved=True,
                old_captures_preserved=True,
                scientific_containers=2,
                moljson_prefix_preserved=True,
            ),
        )
    finally:
        await runner.cleanup()


def entrypoint():
    global args, R, S, DEADLINE
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "output",
        "source",
        "scicode-input",
        "scicode-captures",
        "moljson-input",
        "moljson-result",
        "moljson-request",
        "moljson-repo",
        "h5",
    ]:
        parser.add_argument("--" + name, type=pathlib.Path, required=True)
    parser.add_argument("--port", type=int, default=25200)
    args = parser.parse_args()
    R = args.output.resolve()
    S = args.source.resolve()
    R.mkdir(parents=True, exist_ok=False)
    DEADLINE = time.monotonic() + 1740  # reserve final 60 seconds of outer 1800 for settlement

    asyncio.run(main())


if __name__ == "__main__":
    entrypoint()
