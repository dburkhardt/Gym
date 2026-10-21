# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real pinned QCBench subprocess scoring after native clean-process agent restore.

The historical model response is returned locally; no provider is constructed.
This is an acceptance driver, not a production replay or batch scheduler.
"""

import argparse
import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import psutil
from starlette.responses import JSONResponse, Response

from nemo_gym._checkpoint.agent import AgentSessionParticipant
from nemo_gym._checkpoint.control import CheckpointRequest, CommitRequest, ParticipantControlPlane, RestoreRequest
from nemo_gym.episode_types import EpisodeId
from resources_servers.campaign_creator_scoring.app import Config, CreatorResourcesServer, VerifyRequest
from responses_api_agents.simple_agent.app import SimpleAgentRunRequest
from responses_api_agents.simple_agent.tests.test_app import _make_agent, _mock_response


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2))


async def child(args):
    row = json.loads(args.input.read_text().splitlines()[0])
    body = {k: row[k] for k in ("uuid", "responses_create_params", "verifier_metadata")}
    body.update(_ng_task_index=0, _ng_rollout_index=0, _ng_attempt_index=int(args.phase == "restore"))
    agent, client = _make_agent(False)
    agent._checkpoint_participant = AgentSessionParticipant(agent)
    controller = ParticipantControlPlane(agent.checkpoint_participant, instance_name="agent", lease_grace_seconds=60)
    resource = CreatorResourcesServer(
        config=Config(
            name="resources",
            host="127.0.0.1",
            port=1,
            entrypoint="app.py",
            benchmark="qcbench",
            upstream_repo=str(args.creator.resolve()),
            scoring_timeout_s=600,
        ),
        server_client=client,
    )
    if args.phase == "baseline":
        known = next(
            r
            for r in map(json.loads, args.input.read_text().splitlines())
            if r["reward"] == 1 and not r.get("mask_sample")
        )
        known_result = await resource.verify(VerifyRequest.model_validate(known))
        assert known_result.reward == 1
        write(
            args.output / "known-correct.json", {"uuid": known["uuid"], "result": known_result.model_dump(mode="json")}
        )
    if args.phase == "initial":
        completed_row = json.loads(args.completed_input.read_text().splitlines()[0])
        completed_resource = CreatorResourcesServer(
            config=resource.config.model_copy(
                update={"benchmark": "moljson", "upstream_repo": str(args.completed_creator.resolve())}
            ),
            server_client=client,
        )
        completed_result = await completed_resource.verify(VerifyRequest.model_validate(completed_row))
        write(args.output / "completed-moljson.json", completed_result.model_dump(mode="json"))
        assert not psutil.Process().children(recursive=True), "Completed scorer left children"
    calls, scores = [], []

    def ctl(name="c1", **kw):
        return dict(checkpoint_id=name, deadline_ts=time.time() + 30, **kw)

    async def post(*, server_name, url_path, **kw):
        if server_name == "model":
            if args.phase == "restore":
                raise AssertionError("Restoration must not regenerate the completed response")
            calls.append(kw["json"].model_dump(mode="json") if hasattr(kw["json"], "model_dump") else kw["json"])
            return _mock_response(row["response"])
        if url_path == "/seed_session":
            return _mock_response({})  # unchanged default wait verification; no replay declaration
        if url_path == "/verify":
            result = await resource.verify(VerifyRequest.model_validate(kw["json"]))
            scores.append(result.model_dump(mode="json"))
            return _mock_response(scores[-1])
        req = MagicMock(
            path_params={"rollout_id": url_path.split("/")[2]},
            cookies=kw.get("cookies") or {},
            session={},
            headers=kw.get("headers") or {},
        )
        result = await agent.responses(req, Response(), kw["json"])
        payload = json.loads(result.body) if isinstance(result, JSONResponse) else result.model_dump(mode="json")
        if args.phase == "initial":
            # The completed activation is already held by the native session; checkpoint
            # before /run receives it and before any creator scorer has started.
            await controller.prepare(CheckpointRequest(**ctl()))
            await controller.commit(CommitRequest(**ctl(checkpoint_dir=str(args.output / "checkpoint"))))
            assert not scores
            write(args.output / "ready.json", {"pid": os.getpid(), "model_fixture_requests": calls, "scorer_calls": 0})
            await asyncio.Event().wait()
        return _mock_response(payload)

    client.post = AsyncMock(side_effect=post)
    if args.phase == "restore":
        await controller.restore(
            RestoreRequest(
                **ctl("r1", checkpoint_dir=str(args.output / "checkpoint"), episode_ids=[EpisodeId(rollout_id="0-0")])
            )
        )
        await controller.resume(CheckpointRequest(**ctl("r1")))
    result = await agent.run(MagicMock(cookies={}), SimpleAgentRunRequest.model_validate(body))
    assert not psutil.Process().children(recursive=True), "Scorer left children"
    write(
        args.output / f"{args.phase}.json",
        {
            "pid": os.getpid(),
            "model_fixture_requests": calls,
            "scorer_calls": len(scores),
            "result": result.model_dump(mode="json"),
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--creator", type=Path, required=True)
    p.add_argument("--completed-input", type=Path, required=True)
    p.add_argument("--completed-creator", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--phase", choices=["baseline", "initial", "restore"])
    args = p.parse_args()
    if args.phase:
        asyncio.run(child(args))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--input",
        str(args.input.resolve()),
        "--creator",
        str(args.creator.resolve()),
        "--completed-input",
        str(args.completed_input.resolve()),
        "--completed-creator",
        str(args.completed_creator.resolve()),
        "--output",
        str(args.output.resolve()),
    ]
    subprocess.run(cmd + ["--phase", "baseline"], check=True, timeout=90)
    with (args.output / "initial.log").open("w") as log:
        process = subprocess.Popen(cmd + ["--phase", "initial"], stdout=log, stderr=subprocess.STDOUT)
        try:
            end = time.monotonic() + 60
            while not (args.output / "ready.json").exists():
                if process.poll() is not None:
                    raise RuntimeError("Initial failed; inspect initial.log")
                if time.monotonic() > end:
                    raise TimeoutError("Checkpoint not reached")
                time.sleep(0.05)
            completed_hash = hashlib.sha256((args.output / "completed-moljson.json").read_bytes()).hexdigest()
            process.kill()
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    assert process.returncode == -signal.SIGKILL
    subprocess.run(cmd + ["--phase", "restore"], check=True, timeout=90)
    assert hashlib.sha256((args.output / "completed-moljson.json").read_bytes()).hexdigest() == completed_hash
    baseline = json.loads((args.output / "baseline.json").read_text())
    restored = json.loads((args.output / "restore.json").read_text())
    for key in ["response", "reward", "scoring_details", "mask_sample", "failure_kind"]:
        assert baseline["result"].get(key) == restored["result"].get(key), key
    assert baseline["scorer_calls"] == restored["scorer_calls"] == 1
    assert len(baseline["model_fixture_requests"]) == 1 and not restored["model_fixture_requests"]
    write(
        args.output / "acceptance.json",
        {
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "creator_sha256": hashlib.sha256((args.creator / "eval.py").read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "initial_pid": process.pid,
            "restored_pid": restored["pid"],
            "reward": restored["result"]["reward"],
            "actual_creator_scores": 4,
            "completed_moljson_sha256": completed_hash,
            "completed_input_sha256": hashlib.sha256(args.completed_input.read_bytes()).hexdigest(),
            "owned_children_absent": True,
            "external_model_calls": 0,
            "restored_model_fixture_requests": 0,
            "scope": "Actual creator subprocess scoring after native pre-verifier restoration; no mutable tool-state recovery or in-flight scorer replay.",
        },
    )


if __name__ == "__main__":
    main()
