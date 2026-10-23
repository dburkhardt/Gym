# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned ET evaluator bodies with Harbor-owned execution and file transport."""

import ast
import asyncio
import copy
import hashlib
import importlib.util
import json
import logging
import os
import shlex
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment


DESITTER = "ET_EinsteinInitialData_Exact_metrics_de_Sitter_F"
IMAGE = "sha256:c4884e5e26ca6ba2f0123d9869e4b36ca07a2bcbe7c56ee2b7b9c46fc698ab52"


@dataclass(frozen=True)
class ETRuntime:
    image: str
    thorn: str
    source_filename: str
    references: str
    forbidden: tuple[str, ...]


# Add another task only after its separate image and creator controls qualify.
# The existing de Sitter image contains GRHydro's target and cannot qualify it.
ET_RUNTIMES = MappingProxyType(
    {
        DESITTER: ETRuntime(
            image=IMAGE,
            thorn="EinsteinInitialData/Exact",
            source_filename="metrics/de_Sitter.F",
            references="EinsteinInitialData/Exact/test",
            forbidden=(
                "/opt/Cactus/arrangements/EinsteinInitialData/Exact/src/metrics/de_Sitter.F",
                "/opt/Cactus/configs/sim/build/Exact",
                "/opt/Cactus/configs/sim/build/GRHydro",
                "/opt/Cactus/configs/sim/lib/libthorn_Exact.a",
                "/opt/Cactus/configs/sim/lib/libthorn_GRHydro.a",
                "/opt/Cactus/exe/cactus_sim",
            ),
        ),
    }
)


def checked_et_runtime(raw: dict, image: str) -> ETRuntime:
    runtime = ET_RUNTIMES.get(raw["question_id"])
    if runtime is None or image != runtime.image:
        raise ValueError("ET task runtime is not qualified for this isolated image")
    if raw["content"]["thorn_name"] != runtime.thorn or raw["content"]["src_filename"] != runtime.source_filename:
        raise ValueError("ET task source identity differs from qualified runtime")
    return runtime


def checked_et_source(source: Path) -> dict:
    hashes = json.loads(Path(__file__).with_name("et-source-sha256.json").read_text())
    for name, digest in hashes.items():
        if hashlib.sha256((source / "curation/et" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"ET evaluator source mismatch: {name}")
    return hashes


async def isolate_et(environment: "BaseEnvironment", raw: dict, output: Path) -> None:
    """Require the separately qualified exact isolated image, never delete unknown source."""
    runtime = checked_et_runtime(raw, environment.task_env_config.docker_image)
    forbidden = runtime.forbidden
    result = await environment.exec(" && ".join("test ! -e " + shlex.quote(p) for p in forbidden), timeout_sec=30)
    if result.return_code:
        raise RuntimeError("ET reference isolation failed")
    (output / "isolation.json").write_text(json.dumps({"absent": forbidden, "qualified_image": runtime.image}))


async def score_et(source: Path, environment: "BaseEnvironment", raw: dict, code: str, output: Path) -> dict:
    """Run original ET build/tests/scoring through the existing Harbor environment."""
    hashes = checked_et_source(source)
    runtime = checked_et_runtime(raw, environment.task_env_config.docker_image)
    et = source / "curation/et"
    spec = importlib.util.spec_from_file_location("ainstein_pinned_comparison", et / "smart_diff_comparison.py")
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    tree = ast.parse((et / "ET_evaluator.py").read_text())
    names = {"ConfigManager", "DockerManager", "EinsteinToolkitTester", "EinsteinToolkitScorer"}
    scope = {
        "os": os,
        "json": json,
        "logging": logging,
        "shutil": shutil,
        "compare_directories_with_tolerance": comparison.compare_directories_with_tolerance,
    }
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        + [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(et / "ET_evaluator.py"), "exec"), scope)
    config = copy.deepcopy(json.loads((et / "config.json").read_text()))
    original = config["et"]
    config.update(docker=copy.deepcopy(original["docker"]), dataset={}, prompt={})
    config["evaluation"].update(scoring=original["scoring"], tolerances=original["tolerances"])
    config["paths"].update(original["paths"])
    config["paths"].update(
        thorn_to_tests_file=str(et / "thorn_to_tests.txt"),
        benchmark_base=str(output / "references"),
        test_outputs_base=str(output / "test_outputs"),
        temp_directory=str(output / "temp"),
    )
    (output / "config.json").write_text(json.dumps(config, indent=2))
    # Same creator initialization files and empty generated directory as qualified controls.
    for name, target in (
        ("ThornList", "/opt/Cactus/configs/sim/ThornList"),
        ("run_all_tests.sh", "/opt/Cactus/run_all_tests.sh"),
    ):
        await environment.upload_file(et / name, target)
    setup = await environment.exec(
        "chmod +x /opt/Cactus/run_all_tests.sh && mkdir -p /opt/Cactus/log_files /opt/Cactus/configs/sim/scratch/Formaline",
        timeout_sec=30,
    )
    if setup.return_code:
        raise RuntimeError("ET evaluator setup failed")
    references = output / "references" / runtime.references
    references.parent.mkdir(parents=True, exist_ok=True)
    await environment.download_dir("/opt/Cactus/arrangements/" + runtime.references, references)
    loop = asyncio.get_running_loop()
    cancelled = threading.Event()
    transport_failures = []

    def call(coro):
        if cancelled.is_set():
            coro.close()
            raise RuntimeError("ET evaluator cancelled")
        try:
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
        except BaseException as error:
            transport_failures.append(f"{type(error).__name__}: {error}")
            raise

    class ManagedDocker(scope["DockerManager"]):
        def execute_in_container(self, command: str, working_dir: str | None = None) -> tuple[int, str, str]:
            result = call(
                environment.exec(
                    "cd " + shlex.quote(working_dir or self.working_dir) + " && " + command, timeout_sec=300
                )
            )
            if result.return_code is None or result.return_code < 0 or result.return_code in (124, 137, 143):
                reason = f"ET execution did not settle normally: return_code={result.return_code}"
                transport_failures.append(reason)
                raise RuntimeError(reason)
            with (output / "commands.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "command": command,
                            "return_code": result.return_code,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                        }
                    )
                    + "\n"
                )
            return result.return_code, result.stdout or "", result.stderr or ""

        def copy_to_container(self, src_path: str, dst_path: str) -> bool:
            call(environment.upload_file(Path(src_path), dst_path))
            return True

        def run_command(self, command: str, capture_output: bool = True) -> tuple[int, str, str]:
            args = shlex.split(command)
            if (
                len(args) != 4
                or args[:2] != ["docker", "cp"]
                or not args[2].startswith(self.container_name + ":/opt/Cactus/")
            ):
                raise ValueError("Unexpected ET host command")
            call(environment.download_dir(args[2].split(":", 1)[1], Path(args[3])))
            return 0, "", ""

        def stop_container(self) -> None:
            pass  # Harbor exclusively owns lifecycle; creator destructor cannot stop another container.

    def run():
        settings = scope["ConfigManager"](str(output / "config.json"))
        manager = ManagedDocker(settings, "harbor-owned")
        tester = scope["EinsteinToolkitTester"](settings, manager)
        if not tester.deploy_code(raw["content"]["thorn_name"], raw["content"]["src_filename"], code):
            raise RuntimeError("ET source deployment failed")
        build = tester.build_thorn(raw["content"]["thorn_name"])
        tests = (
            tester.run_tests_for_thorn(raw["content"]["thorn_name"])
            if build["success"]
            else {"total_tests": 0, "passed_tests": 0, "failed_tests": 0, "test_results": []}
        )
        if transport_failures:
            raise RuntimeError("ET transport failure: " + "; ".join(transport_failures))
        return {
            "build": build,
            "tests": tests,
            "native_score": scope["EinsteinToolkitScorer"](settings).calculate_score(build, tests),
            "source_hashes": hashes,
        }

    task = asyncio.create_task(asyncio.to_thread(run))
    try:
        result = await asyncio.shield(task)
    except asyncio.CancelledError:
        cancelled.set()
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if task.done() and not task.cancelled():
            task.exception()
        raise
    finally:
        cancelled.set()
        (output / "transport-status.json").write_text(
            json.dumps({"failures": transport_failures, "worker_settled": task.done()})
        )
    (output / "score.json").write_text(json.dumps(result, indent=2))
    return result
