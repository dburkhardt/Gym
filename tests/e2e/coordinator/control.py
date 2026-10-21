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


async def main(
    out,
    *,
    logged=False,
    completed=False,
    distribution_control=False,
    restart_control=False,
    creator_repo=None,
    rejection_status=429,
):
    blocked_restart = restart_control and rejection_status != 429
    if creator_repo is not None and not restart_control:
        raise ValueError("Creator fixture requires restart control")
    if distribution_control and not completed:
        raise ValueError("Distribution control requires completed-run mode")
    if restart_control:
        completed = True
    task_count = 4 if distribution_control else 3
    repeat_count = 2 if distribution_control else 1
    expected_calls = task_count * repeat_count if completed else 2
    if restart_control and not blocked_restart:
        expected_calls += 1
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
        if restart_control and len(calls) == 2:
            return web.json_response({"error": {"message": "controlled quota rejection"}}, status=rejection_status)
        if len(calls) > 1 and not completed:
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
                        "content": [
                            {
                                "type": "output_text",
                                "text": "\\boxed{A}" if restart_control else "Option 1",
                                "annotations": [],
                            }
                        ],
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
            "resources_servers/campaign_creator_scoring"
            if creator_repo
            else ("resources_servers/mcqa" if restart_control else "resources_servers/proteinlmbench"),
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
            for i in range(task_count)
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
                                "num_repeats": repeat_count,
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
        if restart_control:
            for row in rows:
                row["expected_answer"] = "A"
                row["grading_mode"] = "strict_single_letter_boxed"
            (out / "sample.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            dataset = config.pop("proteinlmbench")["resources_servers"]["proteinlmbench"]["datasets"]
            dataset[0]["name"] = "restart_fixture"
            config["config_paths"] = ["resources_servers/mcqa/configs/mcqa.yaml"]
            config["mcqa_simple_agent"] = {"responses_api_agents": {"simple_agent": {"datasets": dataset}}}
            config["route_failures_to_sidecar"] = True
            config["require_complete"] = False
        if creator_repo is not None:
            config.pop("mcqa_simple_agent")
            config["config_paths"] = ["benchmarks/genomeqa/config.yaml"]
            for row in rows:
                row.pop("expected_answer", None)
                row.pop("grading_mode", None)
                row["uuid"] = row["task_id"]
                row["verifier_metadata"] = {"answer": "A", "options_keys": ["A", "B"]}
            (out / "sample.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            config["genomeqa_resources"] = {
                "resources_servers": {
                    "campaign_creator_scoring": {
                        "upstream_repo": str(creator_repo),
                        "datasets": dataset,
                    }
                }
            }
            config["head_server"]["port"] = 30800
            config["port_range_low"] = 30801
            config["port_range_high"] = 30820
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
            str(task_count),
            "--num-repeats",
            str(repeat_count),
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
            if completed and (child.returncode == 0 or (blocked_restart and child.returncode is not None)):
                break
            if (
                not completed
                and len(calls) == 2
                and (out / "results.jsonl").exists()
                and (out / "results.jsonl").read_text().strip()
            ):
                break
            if child.returncode is not None:
                raise RuntimeError("initial native process terminated early")
            await asyncio.sleep(0.1)
        else:
            raise TimeoutError("did not reach two calls and saved result")
        prior = (out / "results.jsonl").read_bytes()
        (out / "before-results.jsonl").write_bytes(prior)
        shutil.copytree(out / "capture", out / "before-capture")
        descendants = [] if completed else psutil.Process(child.pid).children(recursive=True)
        report["initial_coordinator_pid"] = child.pid
        report["coordinator_was_killed"] = not completed
        report["owned_descendants"] = [p.pid for p in descendants]
        if not completed:
            child.kill()
        await child.wait()
        report["coordinator_exit"] = child.returncode
        report["orphan_cleanup_scope"] = "fresh native CLI reconciles durable exact-owned service receipts"
        if restart_control:
            from nemo_gym.batch_restart import digest

            status_path = out / "results" / "batch_status.json"
            failures_path = out / "results_failures.jsonl"
            failed = [json.loads(line) for line in failures_path.read_text().splitlines()]
            assert len(failed) == 1
            row = failed[0]
            key = f"{row['_ng_task_index']}-{row['_ng_rollout_index']}"
            status = json.loads(status_path.read_text())
            (out / "restart-selection.json").write_text(
                json.dumps(
                    {
                        "execution_binding_sha256": digest(status["execution_binding"]),
                        "failures": [{"capture_key": key, "failure_sha256": digest(row)}],
                    }
                )
            )
            config["guarded_restart_receipt_fpath"] = str(out / "restart-selection.json")
            (out / "runtime.yaml").write_text(yaml.safe_dump(config))
            report["original_failure_sha256"] = hashlib.sha256(failures_path.read_bytes()).hexdigest()
        log2 = (out / "resume.log").open("w")
        resume = await asyncio.create_subprocess_exec(
            *(command + ["--resume"]), env=env, stdout=log2, stderr=asyncio.subprocess.STDOUT
        )
        children.append(resume)
        resume_children = {}
        for _ in range(900):
            try:
                for owned in psutil.Process(resume.pid).children(recursive=True):
                    resume_children[(owned.pid, owned.create_time())] = owned
            except psutil.NoSuchProcess:
                pass
            if resume.returncode is not None or len(calls) > expected_calls:
                break
            await asyncio.sleep(0.1)
        if restart_control and not blocked_restart:
            assert resume.returncode == 0
            completed_bytes = (out / "results.jsonl").read_bytes()
            calls_before_completed_resume = len(calls)
            config.pop("guarded_restart_receipt_fpath")
            (out / "runtime.yaml").write_text(yaml.safe_dump(config))
            with (out / "completed-resume.log").open("w") as completed_log:
                completed_resume = await asyncio.create_subprocess_exec(
                    *(command + ["--resume"]), env=env, stdout=completed_log, stderr=asyncio.subprocess.STDOUT
                )
                children.append(completed_resume)
                assert await asyncio.wait_for(completed_resume.wait(), 60) == 0
            assert len(calls) == calls_before_completed_resume
            assert (out / "results.jsonl").read_bytes() == completed_bytes
            report["completed_resume_additional_calls"] = 0
        report["initial_owned_children_alive"] = [
            p.pid for p in descendants if p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        ]
        report["resume_owned_children_alive"] = [
            p.pid for p in resume_children.values() if p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        ]
        ownership_checks = []
        for receipt_path in (out / "results" / "native_services").glob("*.json"):
            ownership = json.loads(receipt_path.read_text())
            root = ownership.get("root")
            alive = False
            if root:
                try:
                    owned = psutil.Process(root["pid"])
                    alive = owned.create_time() == root["created"] and owned.status() != psutil.STATUS_ZOMBIE
                except psutil.NoSuchProcess:
                    pass
            ownership_checks.append(
                {
                    "receipt": receipt_path.name,
                    "root": root,
                    "cleanup_confirmed": ownership.get("cleanup_confirmed"),
                    "alive": alive,
                }
            )
        report["durable_service_cleanup"] = ownership_checks
        assert ownership_checks and all(
            check["cleanup_confirmed"] and not check["alive"] for check in ownership_checks
        )
        report["capture_evidence_preserved"] = all(
            (out / "capture" / p.name).read_bytes() == p.read_bytes()
            for p in (out / "before-capture").iterdir()
            if p.is_file()
        )
        report["source_hashes"] = {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in [
                "nemo_gym/base_responses_api_model.py",
                "nemo_gym/rollout_collection.py",
                "nemo_gym/cli/eval.py",
                "nemo_gym/cli/env.py",
                "nemo_gym/cli/setup_command.py",
                "nemo_gym/cli/service_ownership.py",
            ]
        }
        report["untouched_task_ids"] = [] if completed else ["fixture-2"]
        report["unrelated_sentinel_survived"] = sentinel.returncode is None
        report["logged_server_launch"] = logged
        report["resume_exit"] = resume.returncode
        report["calls_after_resume"] = len(calls)
        report["completed_result_preserved"] = (
            (out / "results.jsonl").read_bytes().startswith(prior)
            if restart_control
            else (out / "results.jsonl").read_bytes() == prior
        )
        if restart_control and not blocked_restart:
            final_rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
            assert len(final_rows) == task_count
            if creator_repo is not None:
                assert all(
                    row["reward"] == 1 and row["scoring_details"]["metric"] == "creator_option_accuracy"
                    for row in final_rows
                )
            assert sum(row.get("_ng_attempt_index", 0) == 1 for row in final_rows) == 1
            assert hashlib.sha256(failures_path.read_bytes()).hexdigest() == report["original_failure_sha256"]
            report["explicit_new_attempt"] = True
        report["resumed_unknown_resend"] = not completed and len(calls) > 2
        report["completed_run_control"] = completed
        if distribution_control:
            from collections import Counter

            inputs = [
                json.loads(line) for line in (out / "results_materialized_inputs.jsonl").read_text().splitlines()
            ]
            expected = {f"fixture-{i}": repeat_count for i in range(task_count)}
            observed = dict(Counter(row["task_id"] for row in inputs))
            plan = json.loads((out / "results_dispatch_plan.json").read_text())
            assert observed == expected
            assert {row["task_id"]: row["rollouts"] for row in plan["task_distribution"]} == expected
            report["distribution"] = observed
            report["plan_execution_equal"] = True

        report["creator_resource"] = creator_repo is not None
        report["rejection_status"] = rejection_status if restart_control else None
        if blocked_restart:
            assert resume.returncode not in (None, 0)
            assert len(calls) == task_count
            assert (out / "results.jsonl").read_bytes() == prior
            assert "rejected" in (out / "resume.log").read_text() or "unresolved" in (out / "resume.log").read_text()
            report["unknown_response_not_resent"] = True
        report["status"] = (
            "passed"
            if len(calls) == expected_calls
            and (resume.returncode == 0 if completed and not blocked_restart else resume.returncode not in (None, 0))
            and report["completed_result_preserved"]
            and report["capture_evidence_preserved"]
            and not report["initial_owned_children_alive"]
            and (completed or "unresolved provider dispatch" in (out / "resume.log").read_text())
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
