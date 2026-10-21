# SPDX-License-Identifier: Apache-2.0
"""Actual creator/encoder/child/UDS path with local deterministic endpoint; no live inference."""

import argparse, asyncio, json, sys, time
from pathlib import Path
from types import SimpleNamespace
from aiohttp import web
from responses_api_agents.metasyn_native.app import MetaSynAgent, MODEL, native_reward

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gym-model-server",
    action="store_true",
    help="Route through the actual Gym model server and durable provider guard",
)
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


CLEANUPS = []


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
    app.router.add_post(
        "/v1/chat/completions" if args.gym_model_server else "/ng-rollout/control/v1/chat/completions", handler
    )
    runner = web.AppRunner(app)
    CLEANUPS.append(runner.cleanup)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    gym_server = gym_task = None
    endpoint = f"http://127.0.0.1:{port}/ng-rollout/control/v1"
    if args.gym_model_server:
        import socket
        import uvicorn
        from omegaconf import OmegaConf
        from nemo_gym.server_utils import ServerClient, BaseServerConfig
        from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig

        global_config = OmegaConf.create(
            {
                "observability_enabled": True,
                "provider_dispatch_guard": True,
                "model_call_capture_dir": str(args.output / "capture"),
            }
        )
        client = ServerClient(
            head_server_config=BaseServerConfig(name="unused_head", host="127.0.0.1", port=1, entrypoint=""),
            global_config_dict=global_config,
        )
        model = SimpleModelServer(
            config=SimpleModelServerConfig(
                name="policy_model",
                host="127.0.0.1",
                port=1,
                entrypoint="app.py",
                openai_base_url=f"http://127.0.0.1:{port}/v1",
                openai_api_key="local-control-only",
                openai_model=MODEL,
                upstream_max_num_tries=1,
                max_http_attempts=1,
                upstream_request_timeout_seconds=30,
            ),
            server_client=client,
        )
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        sock.setblocking(False)

        async def close_socket():
            sock.close()

        CLEANUPS.append(close_socket)
        gym_port = sock.getsockname()[1]
        gym_server = uvicorn.Server(uvicorn.Config(model.setup_webserver(), log_level="error", lifespan="off"))
        gym_task = asyncio.create_task(gym_server.serve(sockets=[sock]))

        async def close_gym():
            gym_server.should_exit = True
            await asyncio.wait_for(gym_task, 15)
            sock.close()

        CLEANUPS.append(close_gym)
        for _ in range(100):
            if gym_server.started:
                break
            if gym_task.done():
                await gym_task
            await asyncio.sleep(0.02)
        assert gym_server.started
        endpoint = f"http://127.0.0.1:{gym_port}/ng-rollout/control/v1"
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
        out, result = await MetaSynAgent._evaluate(SimpleNamespace(config=cfg), rid, endpoint, "control")
        if args.mode == "judge-disconnect":
            assert result["status"] == "failed"
            assert result.get("error", {}).get("type") == "JudgeError"
            assert "APIConnectionError" in result["error"].get("message", "")
            assert calls[1].get("injected") == "connection aborted after request received"
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
        raise
    finally:
        capture_error = None
        try:
            if args.gym_model_server:
                from nemo_gym.base_responses_api_model import CaptureStore

                store = CaptureStore(args.output / "capture", provider_dispatch_guard=True)
                rows = store.read("control")
                attempts = [a for row in rows for a in row.get("provider_attempts", [])]
                bindings = [row.get("provider_binding", {}).get("logical_call") for row in rows]
                report["gym_capture"] = {
                    "rows": len(rows),
                    "physical_attempts": len(attempts),
                    "outcomes": [a.get("outcome") for a in attempts],
                    "logical_calls": bindings,
                }
                report["scope"] = (
                    "Actual creator, encoder, child, UDS, Gym OpenAI model server, capture middleware and provider-dispatch ledger over loopback HTTP; deterministic provider only; not native CLI or live-model acceptance"
                )
                assert len(attempts) == len(calls), report
                assert len(set(bindings)) == len(bindings), report
                if args.mode == "success":
                    assert len(rows) == 5 and all(a.get("outcome") == "captured" for a in attempts), report
                else:
                    assert (
                        len(attempts) == 2
                        and attempts[0]["outcome"] == "captured"
                        and attempts[1]["outcome"] == "unknown"
                    ), report
        except Exception as error:
            capture_error = error
            report["status"] = "failed"
            report["capture_validation_error"] = repr(error)
        (args.output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("status", "artifacts", "scope")}))
        if capture_error is not None:
            raise capture_error


async def managed_main():
    failure = None
    cleanup_errors = []
    try:
        await main()
    except BaseException as error:
        failure = error
    finally:
        for cleanup in reversed(CLEANUPS):
            try:
                await cleanup()
            except BaseException as error:
                cleanup_errors.append(repr(error))
        path = args.output / "qualification.json"
        report = (
            json.loads(path.read_text())
            if path.exists()
            else {"status": "failed", "scope": "startup did not complete; no acceptance claim"}
        )
        report["cleanup_errors"] = cleanup_errors
        if failure is not None or cleanup_errors:
            report["status"] = "failed"
            if failure is not None:
                report["qualification_error"] = repr(failure)
        path.write_text(json.dumps(report, indent=2) + "\n")
    if failure is not None:
        raise failure
    if cleanup_errors:
        raise RuntimeError("Qualification cleanup failed")


if __name__ == "__main__":
    asyncio.run(managed_main())
