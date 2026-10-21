# SPDX-License-Identifier: Apache-2.0
"""Linux supervisor/collector deadline control; loopback /run only, no model calls."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


async def child(mode, output):
    from aiohttp import web
    from omegaconf import OmegaConf

    from nemo_gym.agent_utils.process_supervisor import collector_deadline_from_environment
    from nemo_gym.config_types import BaseServerConfig
    from nemo_gym.rollout_collection import NG_NO_PERSIST_KEY, RolloutCollectionHelper
    from nemo_gym.server_utils import ServerClient, global_aiohttp_client_exit_async

    deadline = collector_deadline_from_environment()
    if mode == "preflight":
        assert deadline is None
        deadline = time.monotonic() + 4
    assert deadline is not None
    calls = []
    # A scaled operational control: the exact declared task timeout is passed
    # intact. This is not an 1800-second scientific-task runtime qualification.
    task_timeout = 2.0
    row = {
        "agent_ref": {"name": "agent"},
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "task_timeout_seconds": task_timeout,
        "payload": "unchanged synthetic deadline control",
    }

    async def run(request):
        body = await request.json()
        calls.append({"body": body, "started": time.monotonic()})
        assert body == row
        # Finish after admission has closed, but inside the original task budget.
        await asyncio.sleep(0.6)
        calls[-1]["finished"] = time.monotonic()
        return web.json_response({"reward": 1, "fixture": True})

    app = web.Application()
    app.router.add_post("/run", run)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=port),
        global_config_dict=OmegaConf.create(
            {
                "agent": {"responses_api_agents": {"impl": {}}},
                "relay": {
                    "environment_servers": {
                        "legacy_agent": {
                            "agent_server": {"name": "agent"},
                            "host": "127.0.0.1",
                            "port": port,
                        }
                    }
                },
            }
        ),
    )

    # Match native service launch: the shared HTTP client reads this injected
    # configuration rather than parsing the fixture driver argv through Hydra.
    os.environ["NEMO_GYM_CONFIG_DICT"] = OmegaConf.to_yaml(client.global_config_dict)

    class ConnectedCollector(RolloutCollectionHelper):
        def setup_server_client(self, *args, **kwargs):
            return client

    admission_cutoff = deadline - task_timeout
    # Simulated startup consumes real elapsed time inside the actual inherited
    # supervisor envelope, not a patched collector clock or deadline.
    target = admission_cutoff + (0.1 if mode == "late" else -0.3)
    await asyncio.sleep(max(0, target - time.monotonic()))
    began = time.monotonic()
    iterator = ConnectedCollector().run_examples(
        [row],
        dispatch_budget_s=1000,
        drain_margin_s=task_timeout,
    )
    try:
        results = [await future for future in iterator]
    finally:
        await iterator.aclose()
        await global_aiohttp_client_exit_async()
        await runner.cleanup()
    if mode == "late":
        assert calls == [] and results[0][1][NG_NO_PERSIST_KEY]
        assert began > admission_cutoff
    else:
        assert len(calls) == 1 and results[0][1]["reward"] == 1
        assert calls[0]["started"] < admission_cutoff < calls[0]["finished"] < deadline
        assert calls[0]["finished"] - calls[0]["started"] < task_timeout
    write(
        output,
        {
            "mode": mode,
            "pid": os.getpid(),
            "deadline": deadline,
            "admission_cutoff": admission_cutoff,
            "collector_started": began,
            "task_timeout_seconds": task_timeout,
            "calls": calls,
            "results": results,
            "external_model_calls": 0,
            "loopback_listener_closed": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", choices=["late", "timely", "preflight"])
    args = parser.parse_args()
    if args.child:
        asyncio.run(child(args.child, args.output))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    results = []
    for mode in ("late", "timely"):
        receipt = args.output / f"{mode}-supervisor.json"
        output = args.output / f"{mode}.json"
        command = [
            sys.executable,
            str(root / "nemo_gym/agent_utils/process_supervisor.py"),
            "--timeout",
            "15",
            "--cleanup-timeout",
            "2",
            "--collector-cleanup-reserve",
            "2",
            "--receipt",
            str(receipt),
            "--",
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            mode,
            "--output",
            str(output),
        ]
        with (args.output / f"{mode}.log").open("w") as log:
            completed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=25)
        state = json.loads(receipt.read_text())
        assert completed.returncode == 0 and state["return_code"] == 0 and state["cleanup_confirmed"], state
        record = json.loads(output.read_text())
        assert not Path(f"/proc/{record['pid']}").exists()
        results.append({"mode": mode, "command": command, "receipt": state, "child": record})
    write(args.output / "acceptance.json", {"passed": True, "cases": results, "external_model_calls": 0})


if __name__ == "__main__":
    main()
