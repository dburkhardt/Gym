# SPDX-License-Identifier: Apache-2.0
"""Model-free full native Corral CLI control; never a model-performance result.

Requires a separately admitted runtime overlay (existing venvs, ports, ownership,
resource and absolute supervisor bounds). The provider is exclusively loopback.
Use --plan-only first; actual container execution requires explicit --execute.
"""

import argparse
import ast
import asyncio
import hashlib
import json
import os
import signal
import sys
from pathlib import Path

import yaml
from aiohttp import web

from benchmarks.corral.guarded_agent import MODEL
from benchmarks.corral.prepare import stage
from tests.e2e.coordinator.fixture_responses import validated_chat_fixture


ACTIONS = (
    "<action>get_available_functional_groups</action><action_input>{}</action_input>",
    '<action>submit_answer</action><action_input>{"answer":"{}"}</action_input>',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(index):
    if index not in (0, 1):
        raise ValueError("Fixture exhausted: no retries or additional interactions")
    return json.loads(
        validated_chat_fixture(
            {
                "id": f"corral-native-fixture-{index}",
                "object": "chat.completion",
                "created": 1,
                "model": MODEL,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": ACTIONS[index]}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
    )


def compare_state(original, submitted, manifest):
    """Compare every original state entry, not merely submission/score files."""
    entries = manifest["entries"]
    expected = set(entries)
    for root in (original, submitted):
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*")}
        if actual != expected:
            raise ValueError("Incomplete or unexpected creator state tree")
        for name, entry in entries.items():
            p = root / name
            if p.is_symlink() or (p.stat().st_mode & 0o7777) != entry["mode"]:
                raise ValueError("Creator state mode/type mismatch")
            if entry["kind"] == "file":
                if not p.is_file() or p.stat().st_size != entry["size"] or sha(p) != entry["sha256"]:
                    raise ValueError("Creator state bytes mismatch")
            elif entry["kind"] != "directory" or not p.is_dir():
                raise ValueError("Creator state type mismatch")
    if "solver-result.json" not in entries or not any(n.endswith("/commits.sqlite3") for n in entries):
        raise ValueError("Missing submission or SQLite persistence")
    return {n: e["sha256"] for n, e in entries.items() if e["kind"] == "file"}


def expected_observation(creator):
    path = creator / "tasks/retrosynthesis/retrosynthesis/constants.py"
    manifest = json.loads(Path(__file__).with_name("frozen_sources.json").read_text())
    if sha(path) != manifest["files"][path.relative_to(creator).as_posix()]:
        raise ValueError("Changed creator functional-group constants")
    tree = ast.parse(path.read_text())
    smarts = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SMARTS" for t in node.targets)
    )
    return "Observation: " + str(list(ast.literal_eval(smarts).keys()))


def validate_transcript(calls, prompt, observation, submission):
    if len(calls) != 2:
        raise ValueError("Expected exactly two fixture requests")
    first, second = [c["messages"] for c in calls]
    if not any(prompt in str(m.get("content", "")) for m in first):
        raise ValueError("Original task prompt missing")
    if second[:-2] != first or second[-2].get("role") != "assistant" or second[-2].get("content") != ACTIONS[0]:
        raise ValueError("Creator tool-action history changed")
    if (
        second[-1].get("role") != "user"
        or second[-1].get("content") != observation
        or second[-1].get("name") != "get_available_functional_groups"
    ):
        raise ValueError("Creator functional-group tool did not return exact successful result")
    if (
        submission.get("status") != "submitted"
        or submission.get("submission") != "{}"
        or submission.get("output") != {"answer": "{}"}
    ):
        raise ValueError("Submission does not match fixture answer")


def reconcile(out, calls, creator):
    rows = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
    if len(rows) != 1 or rows[0].get("mask_sample") or rows[0].get("reward") != 0:
        raise ValueError("Expected one unmasked canonical zero for deliberate empty submission")
    manifests = list((out / "jobs").glob("**/agent/native-state-manifest.json"))
    if len(manifests) != 1:
        raise ValueError("Expected exactly one creator state manifest")
    manifest_path = manifests[0]
    trial = manifest_path.parent.parent
    manifest = json.loads(manifest_path.read_text())
    container_manifest = json.loads((trial / "agent/native-solver/container-state-manifest.json").read_text())
    if container_manifest != manifest:
        raise ValueError("Container export and uploaded state manifests differ")
    hashes = compare_state(trial / "agent/native-solver/exported-native", trial / "verifier/submitted", manifest)
    submission = json.loads((trial / "agent/native-solver/exported-native/solver-result.json").read_text())
    row = json.loads((out / "single.jsonl").read_text())
    validate_transcript(
        calls, row["verifier_metadata"]["native_task"]["input"]["prompt"], expected_observation(creator), submission
    )
    scorer = json.loads((trial / "verifier/native-scorer/exported-native/scorer-result.json").read_text())
    if scorer.get("commit_hash") != submission.get("commit_hash") or not submission.get("commit_hash"):
        raise ValueError("Scorer evaluated a different submission commit")
    if scorer.get("scorer_version") != "retrosynthesis.score:score_final":
        raise ValueError("Unexpected creator scorer")
    if scorer.get("score") != 0:
        raise ValueError("Canonical scorer did not evaluate deliberate wrong answer as zero")
    captures = [
        json.loads(line) for p in (out / "captures").glob("*.capture.jsonl") for line in p.read_text().splitlines()
    ]
    if len(calls) != 2 or len(captures) != 2 or list((out / "captures").glob("*pending*")):
        raise ValueError("Unexpected provider/capture count or unsettled request")
    if any(c["status_code"] != 200 or len(c["provider_attempts"]) != 1 for c in captures):
        raise ValueError("Provider request failed or retried")
    return {
        "external_model_calls": 0,
        "fixture_calls": 2,
        "state_file_hashes": hashes,
        "canonical_fixture_score": 0,
        "scientific_model_performance": False,
        "cleanup": "Requires independent exact container/scope absence readback",
        "scorer_result": scorer,
    }


async def job_preflight(out, snapshot):
    import importlib

    from harbor.job import Job
    from harbor.models.job.config import JobConfig

    from responses_api_agents.harbor_agent.app import HarborAgent, HarborAgentConfig

    settings = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())["harbor_agent"][
        "responses_api_agents"
    ]["harbor_agent"]
    settings["harbor_datasets"]["corral"]["local_dataset_path"] = str(out / "prepared/tasks")
    for key in ("harbor_agent_kwargs", "harbor_verifier_kwargs"):
        settings[key]["database_snapshot"] = str(snapshot.resolve())
    config = HarborAgentConfig(name="harbor_agent", host="127.0.0.1", port=1, **settings)
    agent = HarborAgent.model_construct(config=config)
    job_config = agent._build_job_config(
        dataset_alias="corral",
        task_name="make_1_lvl3",
        model_name=MODEL,
        api_base="http://127.0.0.1:1/ng-rollout/fixture-preflight/v1",
        job_name="preflight",
        jobs_dir=out / "job-preflight",
        responses_create_params={"input": [], "temperature": 1, "max_output_tokens": 16384},
        gym_provider_dispatch_guard=True,
    )
    job = await Job.create(JobConfig.model_validate(job_config))
    if len(job._trial_configs) != 1:
        raise ValueError("Control must resolve exactly one native trial")
    imports = {}
    for name in (
        "benchmarks.corral.harbor_native",
        "benchmarks.corral.guarded_agent",
        "benchmarks.corral.state_artifact",
        "nemo_gym.state_bundle",
        "responses_api_agents.harbor_agent.app",
        "responses_api_models.openai_model.app",
    ):
        path = Path(importlib.import_module(name).__file__).resolve()
        imports[name] = {"path": str(path), "sha256": sha(path)}
    (out / "job-preflight.json").write_text(
        json.dumps(
            {
                "config": job_config,
                "actual_imports": imports,
                "job_run_called": False,
                "scientific_tasks_modified": False,
            },
            indent=2,
            default=str,
        )
    )


async def run(args):
    runtime = yaml.safe_load(args.runtime.read_text())
    allowed = {
        "head_server",
        "port_range_low",
        "port_range_high",
        "uv_venv_dir",
        "skip_venv_if_present",
        "dispatch_budget_s",
        "drain_margin_s",
        "route_failures_to_sidecar",
    }
    if not isinstance(runtime, dict) or set(runtime) - allowed:
        raise ValueError(
            "Control runtime may only set owned infrastructure; no extra model roles or protocol overrides"
        )
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    prepared = stage(args.creator, out / "prepared")
    rows = [json.loads(x) for x in prepared.read_text().splitlines()]
    selected = next(r for r in rows if r["task_id"] == "make_1_lvl3")
    sample = out / "single.jsonl"
    sample.write_text(json.dumps(selected) + "\n")
    # Validate all fixture bodies before starting even the local responder.
    responses = [fixture(i) for i in range(2)]
    await job_preflight(out, args.snapshot)
    calls = []

    async def provider(request):
        body = await request.json()
        calls.append(body)
        (out / "provider-requests.json").write_text(json.dumps(calls, indent=2))
        if len(calls) > 2:
            raise web.HTTPConflict(text="No fixture replay permitted")
        if body.get("model") != MODEL or body.get("temperature") != 1 or body.get("max_tokens") != 16384:
            raise web.HTTPBadRequest(text="Creator generation protocol changed")
        obj = responses[len(calls) - 1]
        if not body.get("stream"):
            return web.json_response(obj)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for delta, finish in [({"role": "assistant", "content": ACTIONS[len(calls) - 1]}, None), ({}, "stop")]:
            chunk = {
                "id": obj["id"],
                "object": "chat.completion.chunk",
                "created": 1,
                "model": MODEL,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            await response.write(("data: " + json.dumps(chunk) + "\n\n").encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/v1/models", lambda request: web.json_response({"object": "list", "data": [{"id": MODEL}]}))
    app.router.add_post("/v1/chat/completions", provider)
    runner = web.AppRunner(app)
    port = 1  # Planning never starts a listener or contacts any provider.
    if args.execute:
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
    config = {
        "config_paths": ["benchmarks/corral/config.yaml", str(args.runtime.resolve())],
        "policy_model_name": MODEL,
        "policy_api_key": "local-fixture",
        "policy_base_url": f"http://127.0.0.1:{port}/v1",
        "provider_dispatch_guard": True,
        "observability_enabled": True,
        "model_call_capture_dir": str(out / "captures"),
        "use_cached_prepared_benchmarks": True,
        "harbor_agent": {
            "responses_api_agents": {
                "harbor_agent": {
                    "harbor_jobs_dir": str(out / "jobs"),
                    "harbor_datasets": {
                        "corral": {"local_dataset_path": str(out / "prepared/tasks"), "workdir": "/workspace"}
                    },
                    "datasets": [
                        {
                            "name": "corral",
                            "type": "benchmark",
                            "jsonl_fpath": str(sample),
                            "prepare_script": "benchmarks/corral/prepare.py",
                            "num_repeats": 1,
                        }
                    ],
                }
            }
        },
        "policy_model": {
            "responses_api_models": {
                "openai_model": {
                    "openai_base_url": f"http://127.0.0.1:{port}/v1",
                    "openai_api_key": "local-fixture",
                    "openai_model": MODEL,
                    "max_concurrent_requests": 1,
                    "upstream_max_num_tries": 1,
                    "upstream_retry_policy": {"max_attempts": 1},
                }
            }
        },
    }
    config_path = out / "control.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
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
        "--temperature",
        "1",
        "--max-output-tokens",
        "16384",
        "--concurrency",
        "1",
        "--output",
        str(out / "results.jsonl"),
    ]
    if not args.execute:
        command.append("--plan-only")
    (out / "command.json").write_text(json.dumps(command, indent=2))
    (out / "source.json").write_text(
        json.dumps(
            {
                "control": sha(Path(__file__)),
                "runtime": sha(args.runtime),
                "sample": sha(sample),
                "creator_manifest": sha(Path(__file__).with_name("frozen_sources.json")),
                "mode": "execute" if args.execute else "plan-only",
            },
            indent=2,
        )
    )
    env = os.environ.copy()
    env.update(
        CORRAL_SOURCE=str(args.creator.resolve()),
        CORRAL_DATA=str(out / "prepared"),
        CORRAL_DATABASE_SNAPSHOT=str(args.snapshot.resolve()),
    )
    child = None
    try:
        with (out / "cli.log").open("w") as log:
            child = await asyncio.create_subprocess_exec(
                *command, env=env, stdout=log, stderr=asyncio.subprocess.STDOUT
            )
            code = await child.wait()
        (out / "terminal.json").write_text(json.dumps({"return_code": code, "fixture_calls": len(calls)}))
        if code:
            raise RuntimeError("Native CLI failed; preserve all evidence, no automatic rerun")
        if args.execute:
            (out / "acceptance.json").write_text(json.dumps(reconcile(out, calls, args.creator), indent=2))
        elif calls:
            raise RuntimeError("Plan unexpectedly invoked provider")
    finally:
        if child is not None and child.returncode is None:
            child.send_signal(signal.SIGINT)
            await child.wait()  # Outer supported supervisor supplies the owned cleanup bound.
        await runner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--creator", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    asyncio.run(run(parser.parse_args()))
