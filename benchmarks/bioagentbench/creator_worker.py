# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Creator environment entrypoint. No Gym package may be loaded in this process."""

import argparse
import asyncio
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import tomllib
from pathlib import Path


def identity(packet):
    root = Path(packet["creator_root"])
    pins = json.loads(Path(__file__).with_name("source-sha256.json").read_text())
    for relative, digest in pins["creator"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("Creator source drift: " + relative)
    for relative in ("harnesses/opencode", "tasksets/bioagent", "judges/pipeline"):
        sys.path.insert(0, str(root / relative))
    names = (
        "verifiers.v1",
        "bioagent_opencode.harness",
        "bioagent_taskset.taskset",
        "bioagent_taskset.scoring",
        "bioagent_judge.judge",
    )
    modules = {name: importlib.import_module(name) for name in names}
    vfroot = Path(modules["verifiers.v1"].__file__).parent
    for relative, digest in pins["verifiers"].items():
        if hashlib.sha256((vfroot / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("Verifiers source drift: " + relative)
    for name, module in modules.items():
        if name != "verifiers.v1" and not Path(module.__file__).resolve().is_relative_to(root.resolve()):
            raise ValueError("Creator import outside admitted source")
    if (
        hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
        != "f37b0612fe6c8bb13d72dc0a35592509b3b08170c492f1fbd91688edc52aa506"
    ):
        raise ValueError("Creator lock changed")
    lock = tomllib.loads((root / "uv.lock").read_text())
    versions = {p["name"].lower().replace("_", "-"): p["version"] for p in lock["package"] if "version" in p}
    actual = {
        d.metadata["Name"].lower().replace("_", "-"): d.version
        for d in importlib.metadata.distributions()
        if d.metadata.get("Name")
    }
    for name in ("verifiers", "openai", "aiohttp", "anyio", "urllib3", "anthropic"):
        if actual.get(name) != versions.get(name):
            raise ValueError("Creator locked dependency mismatch: " + name)
    for name, version in actual.items():
        if name in versions and version != versions[name]:
            raise ValueError("Installed creator dependency differs from lock: " + name)
    if (
        os.path.abspath(sys.executable) != packet["interpreter"]["path"]
        or hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest() != packet["interpreter"]["sha256"]
    ):
        raise ValueError("Creator interpreter changed")
    from benchmarks.bioagentbench import child_runtime, creator_execution, runtime_channel

    for module in (sys.modules[__name__], creator_execution, child_runtime, runtime_channel):
        path = str(Path(module.__file__).resolve())
        if packet["selected_sources"].get(path) != hashlib.sha256(Path(path).read_bytes()).hexdigest():
            raise ValueError("Child adapter source changed")
    if any(n == "nemo_gym" or n.startswith("nemo_gym.") for n in sys.modules):
        raise ValueError("Gym imported into creator runtime")
    return {
        "creator_lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        "distributions": actual,
        "imports": {
            n: {"path": m.__file__, "sha256": hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()}
            for n, m in modules.items()
        },
        "gym_imported": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    packet = json.loads(args.packet.read_text())
    observed = identity(packet)
    receipts = Path(packet.get("receipt_root", str(args.packet.parent)))
    (receipts / "creator-imports.json").write_text(json.dumps(observed, indent=2) + "\n")
    if args.validate_only:
        return
    from benchmarks.bioagentbench.creator_execution import run_native

    traces = asyncio.run(
        run_native(config=packet["native_config"], packet=packet, output=Path(packet["execution_output"]))
    )
    if any(n == "nemo_gym" or n.startswith("nemo_gym.") for n in sys.modules):
        raise ValueError("Gym imported into creator execution")
    (receipts / "creator-result.json").write_text(json.dumps({"traces": traces}, indent=2) + "\n")


if __name__ == "__main__":
    main()
