# SPDX-License-Identifier: Apache-2.0
"""Actual creator/encoder/child/UDS path with local deterministic endpoint; no live inference."""

import argparse, asyncio, json, sys, time
from pathlib import Path
from types import SimpleNamespace
from aiohttp import web
from responses_api_agents.metasyn_native.app import MetaSynAgent, MODEL, native_reward

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["success", "judge-disconnect"], default="success")
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--references", type=Path, required=True)
parser.add_argument("--creator-source", type=Path, required=True)
parser.add_argument("--python", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
for name in ("data", "references", "creator_source", "python", "output"):
    setattr(args, name, getattr(args, name).absolute())
args.output.mkdir(parents=True, exist_ok=False)
sys.argv = sys.argv[:1]  # Gym lazily parses its own CLI; do not leak probe flags.


async def main():
    data = args.data
    refs = args.references
    review = json.loads(refs.read_text().splitlines()[0])
    rid = review["ID"]
    expected = (data / "rag" / str(rid) / "prompt.txt").read_text()
    calls = []

    async def handler(request):
        p = await request.json()
        assert p["model"] == MODEL
        prompt = p["messages"][-1]["content"]
        if args.mode == "judge-disconnect" and calls:
            calls.append(
                {
                    "ordinal": len(calls),
                    "characters": len(prompt),
                    "logical_call": request.headers.get("x-nemo-gym-logical-call-id"),
                    "injected": "connection aborted after request received",
                }
            )
            request.transport.abort()
            return web.Response(status=503)
        if not calls:
            assert prompt == expected
            content = "# Deterministic test report\nNo articles were included.\nAdults with diagnosed disease. Animal studies excluded. Evidence is limited."
        elif "Extract criteria now:" in prompt:
            content = json.dumps(
                {"inclusion_criteria": ["Adults with diagnosed disease"], "exclusion_criteria": ["Animal studies"]}
            )
        elif "Extract conclusion and insights now:" in prompt:
            content = json.dumps({"conclusion_direction": "Mixed", "key_insights": ["Evidence is limited."]})
        elif "consistency_score" in prompt:
            content = json.dumps({"consistency_score": 0.2, "reason": "Deterministic transport control"})
        elif "structure quality" in prompt:
            content = json.dumps({"score": 2, "reason": "Deterministic transport control"})
        else:
            raise ValueError("Unexpected creator prompt")
        calls.append(
            {
                "ordinal": len(calls),
                "characters": len(prompt),
                "logical_call": request.headers.get("x-nemo-gym-logical-call-id"),
            }
        )
        return web.json_response(
            {
                "id": "local-control",
                "object": "chat.completion",
                "created": 0,
                "model": MODEL,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    app = web.Application(client_max_size=16 * 1024**2)
    app.router.add_post("/ng-rollout/control/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    output = args.output / "attempt"
    cfg = SimpleNamespace(
        output_root=output,
        materialized_root=data,
        reference_file=refs,
        runner_script=Path(__file__).with_name("native_runner.py"),
        native_python=str(args.python),
        creator_source_root=args.creator_source,
        scientific_threads=1,
    )
    report = {"status": "interrupted", "calls": calls, "artifacts": str(output)}
    try:
        out, result = await MetaSynAgent._evaluate(
            SimpleNamespace(config=cfg), rid, f"http://127.0.0.1:{port}/ng-rollout/control/v1", "control"
        )
        if args.mode == "judge-disconnect":
            assert result["status"] == "failed"
            assert len(calls) == 2
            try:
                native_reward(result)
            except ValueError:
                pass
            else:
                raise AssertionError("Failed judge result assigned a reward")
        else:
            assert result["status"] == "succeeded"
        report = {
            "status": "passed",
            "mode": args.mode,
            "calls": calls,
            "evaluation": result,
            "artifacts": str(out),
            "scope": "Actual creator, encoder, managed child, SDK and UDS; deterministic local endpoint replaces Gym model server and provider; not live model acceptance",
        }
    except Exception as e:
        report = {
            "status": "failed",
            "exception": repr(e),
            "calls": calls,
            "artifacts": str(output),
            "scope": "No external inference",
        }
        if args.mode != "judge-disconnect":
            raise
        assert len(calls) == 2, calls
        assert not (output / str(rid) / "evaluation.json").exists()
        assert (output / str(rid) / "failure.json").exists()
        report["status"] = "expected_failure_passed"
        report["verified"] = (
            "One solver call and one ambiguous judge request; creator retries made no further physical requests; no score assigned"
        )
    finally:
        await runner.cleanup()
        (args.output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("status", "artifacts", "scope")}))


if __name__ == "__main__":
    asyncio.run(main())
