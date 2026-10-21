# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline fresh-process control using preserved real SciCode inputs/captures.

No scientific verifier is executed: its exact input is retained and compared.
The deterministic transport returns historical responses, never contacts a provider.
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
from unittest.mock import patch

from nemo_gym._checkpoint.agent import AgentSessionParticipant
from nemo_gym._checkpoint.control import CheckpointRequest, CommitRequest, ParticipantControlPlane, RestoreRequest
from nemo_gym.base_responses_api_model import CaptureStore, _CaptureMiddleware
from nemo_gym.episode_types import EpisodeId
from nemo_gym.provider_dispatch import current_provider_dispatch
from responses_api_agents.scicode_agent.app import ScicodeAgentRunRequest
from responses_api_agents.scicode_agent.tests.test_app import _agent, _FakeRequest, _Resp


def write(path, value):
    path.write_text(json.dumps(value, indent=2))


async def child(args):
    out = args.output
    body = json.loads(args.input.read_text())
    body["_ng_rollout_id"] = "saved-scicode" if args.phase != "restore" else "saved-scicode-a1"
    records = [json.loads(line) for line in args.captures.read_text().splitlines()]
    assert len(records) == len(body["sub_steps"]) == 2
    requests, physical = [], []
    agent = _agent()
    agent.server_client.global_config_dict["provider_dispatch_guard"] = True
    agent._checkpoint_participant = AgentSessionParticipant(agent)
    store = CaptureStore(
        out / ("baseline-capture" if args.phase == "baseline" else "capture"), provider_dispatch_guard=True
    )
    saved = asyncio.Event()
    never = asyncio.Event()

    async def provider(scope, receive, send):
        message = await receive()
        payload = json.loads(message["body"])
        matching = [r for r in records if r["request"] == payload]
        if len(matching) != 1:
            raise AssertionError("Generated request does not equal one preserved scientific request")
        dispatch = current_provider_dispatch.get()
        dispatch.before_send()
        physical.append(payload)
        dispatch.response_headers(200)
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": json.dumps(matching[0]["response"]).encode()})

    middleware = _CaptureMiddleware(
        provider, store=store, model_server_name="policy_model", model_config_digest="b" * 64
    )

    async def post(**kwargs):
        if kwargs["url_path"] == "/verify":
            write(out / f"{args.phase}-verify.json", kwargs["json"])
            return _Resp({"verification_not_executed": True, "solutions": kwargs["json"]["solutions"]})
        requests.append(kwargs)
        messages = [{"type": "http.request", "body": json.dumps(kwargs["json"]).encode()}]
        sent = []

        async def receive():
            return messages.pop(0)

        async def send(message):
            sent.append(message)

        await middleware(
            {
                "type": "http",
                "method": "POST",
                "path": kwargs["url_path"],
                "headers": [(k.encode(), v.encode()) for k, v in kwargs["headers"].items()],
            },
            receive,
            send,
        )
        if args.phase == "initial" and len(requests) == 2:
            saved.set()
            await never.wait()
        return _Resp(json.loads(sent[-1]["body"]))

    agent.server_client.post.side_effect = post
    controller = ParticipantControlPlane(agent.checkpoint_participant, instance_name="agent", lease_grace_seconds=60)

    def ctl(name="c1", **kwargs):
        return dict(checkpoint_id=name, deadline_ts=time.time() + 30, **kwargs)

    with patch("responses_api_agents.scicode_agent.app.raise_for_status", return_value=None):
        if args.phase == "restore":
            await controller.restore(
                RestoreRequest(
                    **ctl(
                        "r1",
                        checkpoint_dir=str(out / "checkpoint"),
                        episode_ids=[EpisodeId(rollout_id="saved-scicode")],
                    )
                )
            )
            await controller.resume(CheckpointRequest(**ctl("r1")))
        task = asyncio.create_task(agent.run(_FakeRequest(), ScicodeAgentRunRequest.model_validate(body)))
        if args.phase == "initial":
            waiter = asyncio.create_task(saved.wait())
            done, _ = await asyncio.wait([task, waiter], return_when=asyncio.FIRST_COMPLETED, timeout=30)
            if task in done:
                await task
                raise AssertionError("Run finished before checkpoint boundary")
            if waiter not in done:
                raise TimeoutError("checkpoint boundary not reached")
            await controller.prepare(CheckpointRequest(**ctl()))
            await controller.commit(CommitRequest(**ctl(checkpoint_dir=str(out / "checkpoint"))))
            [record] = await agent.checkpoint_participant.export(None)
            assert record.episode["next_step"] == 1
            write(
                out / "initial-calls.json",
                {"requests": requests, "physical": physical, "checkpoint_state": record.episode},
            )
            write(out / "ready.json", {"pid": os.getpid()})
            await never.wait()
        else:
            result = await task
            write(out / f"{args.phase}-calls.json", {"requests": requests, "physical": physical, "result": result})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["baseline", "initial", "restore"])
    args = parser.parse_args()
    if args.phase:
        asyncio.run(child(args))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--input",
        str(args.input.resolve()),
        "--captures",
        str(args.captures.resolve()),
        "--output",
        str(args.output.resolve()),
    ]
    subprocess.run(command + ["--phase", "baseline"], check=True, timeout=60)
    with (args.output / "initial.log").open("w") as log:
        process = subprocess.Popen(command + ["--phase", "initial"], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while not (args.output / "ready.json").exists():
                if process.poll() is not None:
                    raise RuntimeError("Initial process failed; inspect initial.log")
                if time.monotonic() > deadline:
                    raise TimeoutError("Initial checkpoint timeout")
                time.sleep(0.05)
            process.kill()
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    assert process.returncode == -signal.SIGKILL
    subprocess.run(command + ["--phase", "restore"], check=True, timeout=60)
    baseline = json.loads((args.output / "baseline-calls.json").read_text())
    initial = json.loads((args.output / "initial-calls.json").read_text())
    restored = json.loads((args.output / "restore-calls.json").read_text())
    assert len(baseline["physical"]) == len(initial["physical"]) == 2 and not restored["physical"]
    assert initial["requests"][1]["headers"] == restored["requests"][0]["headers"]
    assert initial["requests"][1]["json"] == restored["requests"][0]["json"] == baseline["requests"][1]["json"]
    a = json.loads((args.output / "baseline-verify.json").read_text())
    b = json.loads((args.output / "restore-verify.json").read_text())
    # Attempt/trajectory timestamps legitimately differ; scientific state must not.
    for key in ["solutions", "response", "step_usage", "token_usage_version"]:
        assert a[key] == b[key], key
    write(
        args.output / "acceptance.json",
        {
            "task_id": json.loads(args.input.read_text())["problem_id"],
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "capture_sha256": hashlib.sha256(args.captures.read_bytes()).hexdigest(),
            "initial_killed": True,
            "restored_new_physical_calls": 0,
            "scientific_verifier_executed": False,
            "external_inference_calls": 0,
            "next_request_equal": True,
            "scientific_state_equal": True,
        },
    )


if __name__ == "__main__":
    main()
