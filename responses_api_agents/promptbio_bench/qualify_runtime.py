# SPDX-License-Identifier: Apache-2.0
"""Actual pinned creator/Docker/UDS paired control; never contacts an LLM provider."""

import argparse
import asyncio
import hashlib
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

from aiohttp import web

from nemo_gym.sandbox.lifecycle import _publish
from nemo_gym.server_utils import GlobalAIOHTTPAsyncClientConfig, set_global_aiohttp_client
from responses_api_agents.promptbio_bench.managed_runtime import grade_managed, validate_image
from responses_api_agents.promptbio_bench.nemotron_proxy import MODEL


def fixture_reply(payload, fixtures):
    names = [item["function"]["name"] for item in payload.get("tools", [])]
    if len(names) != 1:
        raise ValueError("Unexpected creator response schema")
    name = names[0]
    matches = [item for item in fixtures if item["name"] == name]
    if not matches:
        raise ValueError("Unqualified creator role: " + name)
    arguments = json.loads(json.dumps(matches[0]["arguments"]))
    if name == "FileMappingResult":
        text = "\n".join(str(m.get("content", "")) for m in payload["messages"])
        references = re.findall(r"^  - (/task/ref_answer/[^\n]+)$", text, re.M)
        candidates = re.findall(r"^  - (/runs/[^\n]+)$", text, re.M)
        if len(references) != 1 or len(candidates) != 1:
            raise ValueError("Control expected exactly one reference and candidate")
        arguments["mappings"][0]["reference_file"] = references[0]
        arguments["mappings"][0]["candidate_file"] = candidates[0]
    return name, {
        "id": "chatcmpl-deterministic-control",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_control",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def qualify(args):
    validate_image(args.image)
    task = args.task.resolve()
    if task.name != "b-4-3":
        raise ValueError("This control is qualified only for original frozen b-4-3, not another sample")
    actual = hashlib.sha256((task / "task.json").read_bytes()).hexdigest()
    if actual != args.task_sha256:
        raise ValueError("Pinned task definition checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    references = list((task / "ref_answer").iterdir())
    if len(references) != 1 or not references[0].is_file():
        raise ValueError("Unexpected b-4-3 reference layout")
    runs = []
    for name in ("oracle", "missing"):
        run = args.output / name
        destination = run / "workspace/outputs"
        destination.mkdir(parents=True)
        if name == "oracle":
            shutil.copyfile(references[0], destination / references[0].name)
        _publish(run / "receipt.json", {"control": name, "status": "saved_output_control", "task_sha256": actual})
        runs.append(run)
    fixture_path = Path(__file__).parent / "tests/creator-b43-fixture.json"
    fixture = json.loads(fixture_path.read_text())
    calls = []

    async def provider(request):
        payload = await request.json()
        if payload.get("model") != MODEL or payload.get("stream") or len(calls) >= 12:
            raise web.HTTPBadRequest()
        headers = {
            key: request.headers.get(key) for key in ("x-nemo-gym-logical-call-id", "x-nemo-gym-call-config-sha256")
        }
        if not all(headers.values()):
            raise web.HTTPForbidden()
        try:
            name, reply = fixture_reply(payload, fixture["responses"])
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error))
        calls.append({"schema": name, "headers": headers, "request": payload, "response": reply})
        _publish(args.output / "deterministic-provider.json", {"calls": calls, "live_provider_calls": 0})
        return web.json_response(reply)

    application = web.Application()
    application.router.add_post("/ng-rollout/promptbio-control/v1/chat/completions", provider)
    runner = web.AppRunner(application)
    await runner.setup()
    client = None
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        client = set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
        scores = await grade_managed(
            task,
            runs,
            scorer_dir=args.scorer.resolve(),
            python="python",
            output=args.output / "grading",
            endpoint=f"http://127.0.0.1:{port}/ng-rollout/promptbio-control/v1",
            rollout_id="promptbio-control",
            scorer_image=args.image,
            scorer_memory_mib=2048,
        )
        if scores != [1.0, 0.0]:
            raise AssertionError(f"Creator paired scores changed: {scores}")
        counts = Counter(c["schema"] for c in calls)
        if counts["StrategyRecommendation"] != 1 or counts["FileMappingResult"] != 1:
            raise AssertionError(f"Paired creator sharing changed: {counts}")
        _publish(
            args.output / "complete.json",
            {
                "scores": scores,
                "scorer_memory_mib": 2048,
                "production_memory_mib": 4096,
                "production_placement_qualified": False,
                "schema_counts": dict(counts),
                "local_provider_calls": len(calls),
                "live_provider_calls": 0,
                "task_sha256": actual,
                "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                "qualification": "Actual creator pipeline and shared Docker/UDS guard; deterministic fixture provider, not native Gym model capture or benchmark performance.",
            },
        )
    finally:
        if client is not None:
            await client.close()
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--task-sha256", required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(qualify(parser.parse_args()))


if __name__ == "__main__":
    main()
