# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side Docker solver and unmodified upstream paired evaluator.

Only task.json and data enter the solver container. Neither API keys nor references do.
"""

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4


DATA_REVISION = "078420b69e25c41c99a79135724b6ae6a6c82b69"
SCORER_REVISION = "e2a1947894eac530900247f00c0b726363b540ed"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run bash in the task sandbox.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    }
]


def safe_task_id(task_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
        raise ValueError("Invalid task ID")
    return task_id


def download_task(task_id: str, cache: Path) -> Path:
    """Fetch only a selected task at the pinned dataset revision."""
    from huggingface_hub import snapshot_download

    safe_task_id(task_id)
    root = snapshot_download(
        "promptbio-ai/promptbio-bench-data",
        repo_type="dataset",
        revision=DATA_REVISION,
        allow_patterns=[f"tasks/{task_id}/**"],
        cache_dir=str(cache),
    )
    return Path(root) / "tasks" / task_id


def stage_inputs(task_dir: Path, workspace: Path) -> dict:
    """Allowlist model-visible content; reject symlinks resolving outside its task."""
    task = json.loads((task_dir / "task.json").read_text())
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / "task.json").write_text(json.dumps(task, indent=2))
    for relative in task.get("input_files", []):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "data":
            raise ValueError(f"Input must be under data/: {relative}")
        source = task_dir / path
        # HF snapshot symlinks target its content-addressed cache, so copy bytes, never mount the snapshot.
        if not source.is_file():
            raise FileNotFoundError(source)
        target = workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (workspace / "outputs").mkdir()
    return task


def command(argv: list[str], *, timeout: float = 120) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, errors="replace", timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command {argv[0:2]} failed: {result.stderr[-2000:]}")
    return result.stdout


def completion(base_url: str, api_key: str, payload: dict, *, timeout: float) -> dict:
    """Synchronous host HTTP; callers run this worker outside the Gym event loop."""
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"Inference HTTP {exc.code}") from None
        except (TimeoutError, urllib.error.URLError):
            if attempt == 2:
                raise
        time.sleep(2**attempt)
    raise RuntimeError("Inference retry limit")


def solve(
    task_dir: Path,
    output_root: Path,
    *,
    model: str,
    base_url: str,
    api_key: str,
    image: str,
    max_steps: int = 100,
    max_tokens: int = 16384,
    cpus: int = 2,
    memory: str = "4g",
) -> Path:
    """Run one immutable model/task attempt; resume only matching successful receipts."""
    task_id = safe_task_id(task_dir.name)
    signature = {
        "model": model,
        "task": task_id,
        "dataset": DATA_REVISION,
        "image": image,
        "max_steps": max_steps,
        "max_tokens": max_tokens,
        "cpus": cpus,
        "memory": memory,
    }
    label = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    run = output_root / task_id / label
    receipt = run / "receipt.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous.get("status") == "completed" and previous.get("signature") == signature:
            return run
        raise RuntimeError(f"Previous attempt incomplete at {run}; inspect it before retrying")
    run.mkdir(parents=True, exist_ok=True)
    task = stage_inputs(task_dir, run / "workspace")
    name = "promptbio-" + uuid4().hex
    started = time.monotonic()
    deadline = started + int(task.get("timeout_seconds", 3600))
    messages = [
        {
            "role": "system",
            "content": "Solve the task using shell. Work in /workspace. "
            "Read task.json and data/. Write final requested output files into /workspace/outputs/. "
            "No reference solution is available. Finish with a brief summary.",
        },
        {"role": "user", "content": json.dumps(task)},
    ]
    usage = []
    status, failure = "failed", None
    try:
        command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--network",
                "none",
                "--cpus",
                str(cpus),
                "--memory",
                memory,
                "--pids-limit",
                "256",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "-v",
                f"{(run / 'workspace').resolve()}:/workspace",
                "-w",
                "/workspace",
                image,
                "sleep",
                "infinity",
            ]
        )
        for _ in range(max_steps):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Task timeout")
            result = completion(
                base_url,
                api_key,
                {"model": model, "messages": messages, "tools": TOOLS, "temperature": 0, "max_tokens": max_tokens},
                timeout=min(600, remaining),
            )
            with (run / "model-responses.jsonl").open("a") as receipt_stream:
                receipt_stream.write(json.dumps({"request": {"model": model, "messages": messages, "tools": TOOLS, "temperature": 0, "max_tokens": max_tokens}, "response": result}) + "\n")
            if model == "nvidia/nvidia/nemotron-3.5-super-vl" and result.get("model") != model:
                (run / "unexpected-model-response.json").write_text(json.dumps(result, indent=2))
                raise ValueError("Unexpected returned solver model")
            usage.append(result.get("usage"))
            message = result["choices"][0]["message"]
            messages.append(message)
            (run / "trajectory.json").write_text(json.dumps(messages, indent=2))
            calls = message.get("tool_calls") or []
            if model == "nvidia/nvidia/nemotron-3.5-super-vl" and result["choices"][0].get("finish_reason") == "length":
                raise TimeoutError("Solver output token budget exhausted")
            if not calls:
                status = "completed"
                break
            for call in calls:
                try:
                    if call["function"]["name"] != "shell":
                        raise ValueError("Unknown tool")
                    script = json.loads(call["function"]["arguments"])["command"]
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Task timeout")
                    # Container-side timeout prevents timed-out commands continuing in the background.
                    value = command(
                        [
                            "docker",
                            "exec",
                            name,
                            "timeout",
                            str(max(1, int(min(300, remaining)))),
                            "bash",
                            "-lc",
                            script,
                        ],
                        timeout=min(310, remaining + 1),
                    )
                except (ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    value = str(exc)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": value[-30000:]})
        else:
            raise TimeoutError("Agent step limit")
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        receipt.write_text(
            json.dumps(
                {
                    "signature": signature,
                    "status": status,
                    "failure": failure,
                    "elapsed_seconds": time.monotonic() - started,
                    "usage": usage,
                },
                indent=2,
            )
        )
        (run / "trajectory.json").write_text(json.dumps(messages, indent=2))
    return run


def parse_score(path: Path) -> float:
    """Preserve upstream scored-file mean; reject missing/failed grading rather than inventing zero."""
    result = json.loads(path.read_text())
    score = result.get("avg_similarity")
    if (
        isinstance(score, bool)
        or not isinstance(score, (float, int))
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise ValueError(f"No valid upstream score: {result.get('error')}")
    return float(score)


def grade(
    task_dir: Path,
    runs: list[Path],
    *,
    scorer_dir: Path,
    python: str,
    output: Path,
    judge_model: str = "gpt-5.4",
    judge_key_file: Path,
    budget_ledger: Path,
    budget_usd: float = 100,
    judge_base_url: str | None = None,
) -> list[float]:
    """Grade all model outputs in one upstream invocation (shared Detect/Recommend)."""
    if judge_base_url is not None:
        from responses_api_agents.promptbio_bench.nemotron_proxy import MODEL, GATEWAY
        if judge_model != MODEL or judge_base_url.rstrip("/") != GATEWAY:
            raise ValueError("Campaign judge must use exact Super and gateway")
    if command(["git", "-C", str(scorer_dir), "rev-parse", "HEAD"]).strip() != SCORER_REVISION:
        raise ValueError("Evaluator must be at pinned upstream revision")
    if command(["git", "-C", str(scorer_dir), "status", "--porcelain", "--untracked-files=no"]).strip():
        raise ValueError("Evaluator has tracked modifications")
    for run in runs:
        for path in (run / "workspace/outputs").rglob("*"):
            if path.is_symlink():
                raise ValueError("Solver output symlinks are not allowed")
    output.mkdir(parents=True, exist_ok=True)
    labels = [run.name for run in runs]
    paths = [output / f"{task_dir.name}_{label}.json" for label in labels]
    signature = {
        "judge_model": judge_model,
        "judge_base_url": judge_base_url,
        "scorer": SCORER_REVISION,
        "dataset": DATA_REVISION,
        "run_receipts": [hashlib.sha256((run / "receipt.json").read_bytes()).hexdigest() for run in runs],
    }
    receipt = output / "grading-receipt.json"
    if receipt.exists():
        if json.loads(receipt.read_text()) != signature:
            raise ValueError("Grading receipt does not match judge, scorer, dataset, or solver attempts")
        return [parse_score(path) for path in paths]
    if any(path.exists() for path in paths):
        raise ValueError("Partial/unreceipted grading exists; inspect before repeating paid calls")
    if not receipt.exists():
        argv = [
            python,
            str(scorer_dir / "run_eval.py"),
            "--task-dir",
            str(task_dir),
            "--result-dir",
            ",".join(str(run / "workspace/outputs") for run in runs),
            "--label",
            ",".join(labels),
            "--output-dir",
            str(output),
            "--model",
            judge_model,
        ]
        if judge_base_url is not None:
            argv = [python, str(Path(__file__).with_name("campaign_scorer.py")), *argv[1:]]
        from responses_api_agents.promptbio_bench.budget_proxy import proxy

        if judge_base_url is not None:
            from responses_api_agents.promptbio_bench.nemotron_proxy import proxy as campaign_proxy
            context = campaign_proxy(key_file=judge_key_file, ledger=budget_ledger, model=judge_model, base_url=judge_base_url)
        else:
            context = proxy(key_file=judge_key_file, ledger=budget_ledger, limit=budget_usd, model=judge_model)
        with context as url:
            env = {**os.environ, "OPENAI_API_KEY": "metered-local-proxy", "OPENAI_BASE_URL": url}
            with (output / "evaluator.log").open("w") as log:
                subprocess.run(
                    argv, cwd=scorer_dir, env=env, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=3600
                )
    if judge_base_url is not None and budget_ledger.exists():
        requests = [json.loads(line) for line in budget_ledger.read_text().splitlines()]
        if any("failure" in request for request in requests):
            raise ValueError("Campaign judge failure retained in receipts; raw scorer outputs are not valid performance")
    scores = [parse_score(path) for path in paths]
    receipt.write_text(json.dumps(signature, indent=2))
    return scores
