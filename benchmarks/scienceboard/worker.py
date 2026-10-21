# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One trusted creator task through its original Tester and shared transport/lifecycle."""

import argparse
import hashlib
import importlib
import json
import os
import time
from pathlib import Path

from benchmarks.scienceboard.guarded_model import CapturedModelActionError, CreatorModelHTTP, bind_creator_model
from benchmarks.scienceboard.managed_desktop import creator_desktop_factory
from benchmarks.scienceboard.prepare import MANIFEST, checked_source
from benchmarks.scienceboard.qualification import runtime_identity
from benchmarks.scienceboard.result_mapping import read_creator_result
from nemo_gym.sandbox.guarded_chat_transport import publish


MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


def run(packet: dict) -> dict:
    """Run one exact task; callers must admit scientific image and owned cleanup first."""
    for filename, expected in packet["implementation"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError("Selected implementation changed before worker startup")
    loaded = [__file__]
    for name in (
        "benchmarks.scienceboard.guarded_model",
        "benchmarks.scienceboard.managed_desktop",
        "benchmarks.scienceboard.prepare",
        "benchmarks.scienceboard.result_mapping",
        "nemo_gym.sandbox.guarded_chat_transport",
        "nemo_gym.guarded_chat_transport",
    ):
        loaded.append(importlib.import_module(name).__file__)
    for filename in loaded:
        actual = Path(filename).resolve()
        if packet["implementation"].get(str(actual)) != hashlib.sha256(actual.read_bytes()).hexdigest():
            raise ValueError("Actual worker import does not match selected implementation")
    source = Path(packet["source"])
    out = Path(packet["output"])
    manifest = json.loads(MANIFEST.read_text())
    item = next(item for item in manifest["tasks"] if item["task_id"] == packet["task_id"])
    task = checked_source(source, item["path"], item["sha256"])
    creator = importlib.import_module("sci")
    if Path(creator.__file__).resolve() != source.resolve() / "sci/__init__.py":
        raise ValueError("Creator import is not the pinned selected source")
    model_module = importlib.import_module("sci.base.model")
    docker_module = importlib.import_module("sci.docker")
    tester_module = importlib.import_module("sci.Tester")
    factory = creator_desktop_factory(
        source,
        metadata=packet["image"],
        audit=packet["audit"],
        resources=packet["resources"],
        receipt_dir=out / "desktops",
    )
    actual_runtime = runtime_identity()
    if packet.get("mode") != "qualification" and actual_runtime != packet["expected_runtime"]:
        raise ValueError("Native interpreter dependencies differ from qualified runtime")
    imports = {
        m.__name__: {"path": m.__file__, "sha256": hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()}
        for m in (creator, model_module, docker_module, tester_module)
    }
    publish(out / "creator-imports.json", imports)
    publish(out / "runtime.json", actual_runtime)
    transport = CreatorModelHTTP(
        socket_path=packet["socket"],
        token=os.environ["SCIENCEBOARD_BRIDGE_TOKEN"],
        lineage=packet["lineage"],
        digest=packet["digest"],
        model=MODEL,
        output=out / "calls",
    )
    original_desktop, original_call = docker_module.DockerDesktopEnv, tester_module.TaskInfo.__call__
    failures = []
    tester = None
    result_path = out / "missing-result.out"

    def observed_call(info):
        try:
            return original_call(info)
        except CapturedModelActionError:
            # Typed captured model failure must not be reclassified as a creator exception.
            raise
        except BaseException as error:
            failures.append({"type": type(error).__name__, "phase": "creator_task"})
            raise

    try:
        docker_module.DockerDesktopEnv = factory
        tester_module.TaskInfo.__call__ = observed_call
        with bind_creator_model(model_module, source, transport):
            # Match the pinned main.py key_bank screenshot AIO baseline. Sampling,
            # reasoning instructions, screenshot and per-task step count stay native.
            agent = creator.Automata(
                model_style="openai",
                base_url="http://gym-internal/chat/completions",
                model_name=MODEL,
                api_key=None,
                top_p=None,
                temperature=0,
                overflow_style="openai_gpt",
                mirror_system_to_user=True,
            )(creator.AIOAgent)
            tester = creator.Tester(
                tasks_path=str(task),
                logs_path=str(out / "creator_logs"),
                community=creator.AllInOne(agent),
                obs_types={creator.OBS.screenshot},
                vm_path=packet["image"]["Id"],
                headless=True,
                ignore=False,
                handle_managers=creator.Presets.spawn_docker_managers,
            )
            if len(tester.task_info) != 1:
                raise ValueError("Creator skipped or expanded the single frozen task")
            result_path = out / "creator_logs" / tester.task_info[0].ident / "result.out"
            if not result_path.resolve().is_relative_to(out.resolve()):
                raise ValueError("Creator result path escaped owned output")
            if packet.get("mode") == "qualification":
                info = tester.task_info[0]
                with info.task.manager:
                    if not info.task.init():
                        raise RuntimeError("Exact creator initializer failed qualification")
                    publish(
                        out / "initialized.json",
                        {"task_id": item["task_id"], "task_sha256": item["sha256"], "pid": os.getpid()},
                    )
                    while True:
                        time.sleep(1)  # Parent intentionally SIGKILLs owned worker after this receipt.
            else:
                tester()
    except CapturedModelActionError:
        pass
    except BaseException as error:
        failures.append({"type": type(error).__name__, "phase": "tester_or_cleanup"})
    finally:
        if tester is not None:
            for manager in tester.managers.values():
                if manager.entered:
                    try:
                        manager.__exit__(None, None, None)
                    except BaseException as error:
                        failures.append({"type": type(error).__name__, "phase": "owned_cleanup"})
        try:
            transport.close()
        except BaseException as error:
            failures.append({"type": type(error).__name__, "phase": "transport_cleanup"})
        finally:
            docker_module.DockerDesktopEnv = original_desktop
            tester_module.TaskInfo.__call__ = original_call
    result = read_creator_result(
        result_path,
        task_id=item["task_id"],
        transport_outcome=transport.outcome,
        creator_error=json.dumps(failures) if failures else None,
    )
    result.update(protocol=packet["protocol"], task_sha256=item["sha256"], failures=failures)
    publish(out / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.packet.read_text()))


if __name__ == "__main__":
    main()
