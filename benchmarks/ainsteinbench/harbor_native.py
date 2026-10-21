# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Creator-compatible async four-tool loop on Harbor-owned environments."""

import asyncio
import hashlib
import json
import shlex
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from harbor.agents.base import BaseAgent
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from benchmarks.ainsteinbench.native import MODEL
from benchmarks.ainsteinbench.scoring import qualification
from benchmarks.ainsteinbench.source import checked_source, creator_methods, parse_log, prompt
from benchmarks.ainsteinbench.trajectory import publish as publish_trajectory
from nemo_gym.provider_dispatch import configuration_digest
from nemo_gym.sandbox.lifecycle import _publish
from nemo_gym.server_utils import request


class NativeIterationLimitError(RuntimeError):
    pass


class NativeNoSolutionError(RuntimeError):
    pass


async def execute_tool(environment, name, arguments):
    try:
        return await _execute_tool(environment, name, arguments)
    except KeyError:
        raise
    except Exception as error:
        # Creator DockerAgent returns tool execution errors to the model.
        # Cancellation remains BaseException and propagates to Harbor cleanup.
        return f"Error: {error}"


async def _execute_tool(environment, name, arguments):
    if name == "write_file":
        with TemporaryDirectory() as directory:
            file = Path(directory) / "content"
            file.write_text(arguments["content"])
            await environment.upload_file(file, arguments["path"])
        return "File written successfully"
    if name in ("exec_command", "read_file"):
        command = arguments["command"] if name == "exec_command" else f"cat {arguments['path']}"
        # Preserve Docker SDK's shlex interpretation of the creator command.
        try:
            argv = shlex.split(f"bash -c '{command}'")
        except ValueError as error:
            return f"Error: {error}"
        result = await environment.exec(shlex.join(argv))
        if name == "read_file":
            return (
                result.stdout
                if result.return_code == 0
                else f"Error: File not found or cannot be read: {result.stderr or ''}"
            )
        return f"Exit code: {result.return_code}\nStdout:\n{result.stdout or ''}\nStderr:\n{result.stderr or ''}"
    return f"Unknown tool: {name}"


async def creator_loop(*, raw, source, environment, completion, save):
    methods = creator_methods(source)
    messages = [
        {"role": "system", "content": prompt(source, raw)},
        {
            "role": "user",
            "content": "Please solve this problem. Start by exploring the environment and understanding the task.",
        },
    ]
    solution = None
    for ordinal in range(30):
        response = await completion(ordinal, messages, methods["TOOLS"])
        message = response["choices"][0]["message"]
        messages.append(message)
        save(messages)
        if message.get("content") and not message.get("tool_calls"):
            solution = message["content"]
            break
        if message.get("tool_calls"):
            for tool in message["tool_calls"]:
                name = tool["function"]["name"]
                arguments = json.loads(tool["function"]["arguments"])
                if name == "finish":
                    solution = arguments["solution"]
                    break
                result = await execute_tool(environment, name, arguments)
                messages.append({"role": "tool", "tool_call_id": tool["id"], "content": result[:10000]})
                save(messages)
            if solution:
                break
        else:
            break
    if not solution:
        error = NativeIterationLimitError if ordinal == 29 else NativeNoSolutionError
        raise error("Agent did not produce a solution within iteration limit")
    return methods["_extract_code_from_markdown"](None, solution)


async def isolate(environment, raw, output):
    command = "git rev-parse HEAD"
    result = await environment.exec(command, timeout_sec=30)
    if result.return_code or result.stdout.strip() != raw["content"]["base_commit"]:
        raise RuntimeError("AInstein base commit mismatch")
    # Hash symlink targets without following them, matching the repaired pilot.
    script = "import subprocess,pathlib,hashlib,json,os; fs=subprocess.check_output(['git','ls-files','-z']).decode().rstrip('\\0').split('\\0'); print(json.dumps({f:({'symlink':os.readlink(f)} if pathlib.Path(f).is_symlink() else {'sha256':hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()}) for f in fs},sort_keys=True))"
    before = await environment.exec("python -c " + shlex.quote(script), timeout_sec=30)
    if before.return_code:
        raise RuntimeError("AInstein tracked source hash failed")
    command = 'rm -f /home/fix.patch /home/test.patch /home/fix-run.sh /home/test-run.sh /home/run.sh; rm -rf .git; git init -q; git -c user.name=Benchmark -c user.email=benchmark@local add -A; git -c user.name=Benchmark -c user.email=benchmark@local commit -qm base; test ! -e /home/fix.patch && test ! -e /home/test.patch && test ! -e /home/fix-run.sh && test ! -e /home/test-run.sh && test ! -e /home/run.sh && test "$(git rev-list --count HEAD)" = 1'
    result = await environment.exec(command, timeout_sec=90)
    after_script = script.replace(
        "fs=subprocess.check_output(['git','ls-files','-z']).decode().rstrip('\\0').split('\\0')",
        "fs=" + repr(list(json.loads(before.stdout))),
    )
    after = await environment.exec("python -c " + shlex.quote(after_script), timeout_sec=30)
    if result.return_code or after.return_code or before.stdout != after.stdout:
        raise RuntimeError("AInstein reference/history isolation failed")
    _publish(output / "base-files.json", json.loads(before.stdout))
    _publish(
        output / "isolation.json",
        {
            "base_commit": raw["content"]["base_commit"],
            "tracked_bytes_unchanged": True,
            "history_commits": 1,
            "evaluator_files_absent": True,
        },
    )


class AInsteinAgent(BaseAgent):
    def __init__(self, *, source, api_base, gym_provider_dispatch_guard: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.source = Path(source)
        self.endpoint = api_base
        if self.model_name != MODEL or "/ng-rollout/" not in api_base or gym_provider_dispatch_guard is not True:
            raise ValueError("AInstein requires guarded native exact-model endpoint")

    @staticmethod
    def name():
        return "ainstein-creator-four-tools"

    def version(self):
        return "d9b1383e86c2ae43dcb3ddbcaf34c21ceb786cca"

    async def setup(self, environment):
        pass

    async def run(self, instruction, environment, context):
        rows, hashes = checked_source(self.source)
        raw = rows[instruction.strip()]
        if raw["question_type"] != "multi_swe_bench":
            raise RuntimeError("ET source/runtime qualification blocked; no model dispatch permitted")
        output = self.logs_dir / "native"
        output.mkdir(parents=True, exist_ok=False)
        lineage = uuid.uuid4().hex
        digest = configuration_digest(
            {
                "source": hashes,
                "task": raw,
                "model": MODEL,
                "max_tokens": 16384,
                "iterations": 30,
                "adapter_sha256": {
                    name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ("harbor_native.py", "source.py", "native.py", "trajectory.py", "scoring.py")
                },
            }
        )
        _publish(
            output / "identity.json",
            {
                "lineage": lineage,
                "configuration_sha256": digest,
                "task_id": raw["question_id"],
                "resume": "full shell continuation unsupported; existing attempt fails closed",
            },
        )
        await isolate(environment, raw, output)

        async def completion(ordinal, messages, tools):
            payload = {
                "model": MODEL,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": 16384,
            }
            headers = {
                "x-nemo-gym-logical-call-id": f"ainstein:{lineage}:{ordinal}",
                "x-nemo-gym-call-config-sha256": digest,
            }
            _publish(output / f"call-{ordinal}.intent.json", {"headers": headers, "payload": payload})
            publish_trajectory(output, messages, context)
            async with asyncio.timeout(600):
                response = await request(
                    "POST",
                    self.endpoint.rstrip("/") + "/chat/completions",
                    json=payload,
                    headers=headers,
                    _max_num_tries=1,
                    _max_connection_retries=0,
                    _resend_after_disconnect=False,
                )
                try:
                    response.raise_for_status()
                    body = await response.json()
                finally:
                    response.release()
            _publish(output / f"call-{ordinal}.response.json", body)
            return body

        try:
            answer = await creator_loop(
                raw=raw,
                source=self.source,
                environment=environment,
                completion=completion,
                save=lambda messages: (
                    _publish(output / "messages.json", messages),
                    publish_trajectory(output, messages, context),
                ),
            )
        except NativeIterationLimitError:
            _publish(
                output / "outcome.json",
                {"status": "model_budget_exhausted", "attempted": True, "score_available": False},
            )
            raise
        except NativeNoSolutionError:
            _publish(
                output / "outcome.json", {"status": "model_no_solution", "attempted": True, "score_available": False}
            )
            raise
        except BaseException:
            _publish(
                output / "outcome.json",
                {"status": "infrastructure_or_protocol_failure", "attempted": True, "score_available": False},
            )
            raise
        patch = output / "answer.patch"
        patch.write_text(answer)
        result = await environment.exec("mkdir -p /submission", timeout_sec=30)
        if result.return_code:
            raise RuntimeError("Submission directory unavailable")
        await environment.upload_file(patch, "/submission/answer.patch")
        _publish(
            output / "outcome.json",
            {
                "status": "submitted",
                "task_id": raw["question_id"],
                "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
            },
        )
        await environment.upload_file(output / "outcome.json", "/submission/outcome.json")


class AInsteinVerifier(BaseVerifier):
    def __init__(self, *, source, **kwargs):
        super().__init__(**kwargs)
        self.source = Path(source)

    async def verify(self):
        if self.task.config.verifier.environment_mode != "separate":
            raise ValueError("Separate evaluator required")
        rows, hashes = checked_source(self.source)
        raw = rows[self.task.paths.task_dir.name]
        if raw["question_type"] != "multi_swe_bench":
            raise RuntimeError("ET native scorer blocked")
        output = self.trial_paths.verifier_dir / "native"
        output.mkdir(parents=True, exist_ok=False)
        await self.environment.download_dir("/submission", output / "submitted")
        import stat

        for file in (output / "submitted").rglob("*"):
            mode = file.lstat().st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError("Unsafe submitted artifact")
        status = json.loads((output / "submitted/outcome.json").read_text())
        patch = output / "submitted/answer.patch"
        if (
            status.get("status") != "submitted"
            or status.get("task_id") != raw["question_id"]
            or hashlib.sha256(patch.read_bytes()).hexdigest() != status.get("patch_sha256")
        ):
            raise ValueError("Incomplete or wrong-task submitted artifact")
        await self.environment.upload_file(patch, "/home/fix.patch")
        result = await self.environment.exec("bash /home/fix-run.sh", timeout_sec=3500)
        _publish(
            output / "execution.json",
            {"return_code": result.return_code, "stdout": result.stdout, "stderr": result.stderr},
        )
        parsed = parse_log(self.source, (result.stdout or "") + (result.stderr or ""))
        total = parsed["passed_count"] + parsed["failed_count"]
        score = parsed["passed_count"] / total * 100 if total else 0.0
        validity = qualification(
            raw["question_id"], parsed, (result.stdout or "") + (result.stderr or ""), result.return_code
        )
        _publish(
            output / "score.json",
            {
                "task_id": raw["question_id"],
                "overall_score": score,
                "test_result": parsed,
                "total_count": total,
                "success": parsed["failed_count"] == 0 and parsed["passed_count"] > 0,
                "source_hashes": hashes,
                "scorer_adapter_sha256": {
                    name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ("harbor_native.py", "source.py", "scoring.py")
                },
                "qualification": validity,
            },
        )
        if not validity["valid"]:
            raise RuntimeError("Unscored native test execution: " + validity["kind"] + "; raw creator score retained")
        return VerifierResult(rewards={"score": score, "tests_passed": parsed["passed_count"], "tests_total": total})
