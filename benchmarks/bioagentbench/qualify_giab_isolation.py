"""GIAB setup/mount qualification only: no harness, provider, judge or hap.py.

Run under the existing process supervisor; the child uses the existing Runtime
channel and pinned creator environment. This is not scientific acceptance.
"""

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


IMAGE = "sha256:da9808d329a71cb42b2260f91831c3d563597d2afaaf8ff7ac19f325ce5efeb1"


def check_mounts(inspected, *, workspace, source):
    """Check the complete Docker mount list, not just guessed forbidden paths."""
    expected = {
        str(workspace): "/app",
        str(source / "data"): str(source / "data"),
        str(source / "reference"): str(source / "reference"),
    }
    mounts = inspected["Mounts"]
    if (
        len(mounts) != len(expected)
        or any(m.get("Type") != "bind" or expected.get(m.get("Source")) != m.get("Destination") for m in mounts)
        or {m["Source"] for m in mounts} != set(expected)
    ):
        raise ValueError("Actual mount tree differs from the exact public/workspace allowlist")
    if inspected["Image"] != IMAGE:
        raise ValueError("Actual image differs")
    host = inspected["HostConfig"]
    if host["NetworkMode"] != "bridge" or host["Memory"] != 8 * 1024**3:
        raise ValueError("Actual native runtime limits/network differ")
    if host.get("NanoCpus", 0) != 2_000_000_000 and not (
        host.get("CpuPeriod", 0) > 0 and host.get("CpuQuota") == 2 * host["CpuPeriod"]
    ):
        raise ValueError("Actual CPU limit differs")
    return mounts


async def setup_only(task, runtime):
    if task.data.task_id != "giab" or task.data.staging != "local" or not task.data.include_reference:
        raise ValueError("Only the unchanged GIAB local setup is qualified")
    await task.setup(None, runtime)


def trusted_truth(packet):
    """Recheck evaluator bytes inside the actual clean creator process."""
    manifest = Path(packet["asset_manifest"])
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != packet["asset_manifest_sha256"]:
        raise ValueError("Child evaluator manifest changed")
    files = json.loads(manifest.read_text())["tasks"]["giab"]["files"]
    truth = {name: digest for name, digest in files.items() if name.startswith("results/")}
    if not truth:
        raise ValueError("GIAB evaluator inventory is missing")
    checked = {}
    for root in (Path(packet["original_data_root"]), Path(packet["native_config"]["taskset"]["data_root"])):
        for name, expected in truth.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe evaluator path")
            path = root / "giab" / relative
            if path.is_symlink() or any(p.is_symlink() for p in path.parents) or not path.is_file():
                raise ValueError("Evaluator is not a regular readable file")
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError("Child evaluator bytes changed")
            checked[str(path)] = actual
    return checked


async def child(packet, output):
    from benchmarks.bioagentbench.creator_worker import identity

    observed = identity(packet)
    observed["evaluator_files"] = trusted_truth(packet)
    from bioagent_taskset.taskset import BioAgentConfig, BioAgentTaskset

    from benchmarks.bioagentbench.runtime_channel import RuntimeClient

    config = BioAgentConfig.model_validate(packet["native_config"]["taskset"])
    tasks = BioAgentTaskset(config).load()
    if len(tasks) != 1:
        raise ValueError("Qualification must select exactly one GIAB task")
    client = RuntimeClient(Path(packet["runtime_socket"]), packet["ipc_token"], deadline=packet["deadline"])

    class Remote:
        async def run(self, argv, env):
            return SimpleNamespace(**await client.call("run", argv=argv, env=env))

    await setup_only(tasks[0], Remote())
    (output / "creator-setup.json").write_text(
        json.dumps({"identity": observed, "task": tasks[0].data.model_dump(mode="json")}, indent=2)
    )


async def command(argv, *, env=None, timeout=60):
    process = await asyncio.create_subprocess_exec(
        *argv, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.communicate()
        raise
    if process.returncode:
        raise RuntimeError(f"Qualification command failed ({process.returncode}): {stderr.decode()[:2000]}")
    return stdout.decode()


async def qualify(packet, output):
    deadline = time.time() + 540
    from benchmarks.bioagentbench.runtime import runtime_class
    from benchmarks.bioagentbench.runtime_channel import RuntimeServer
    from benchmarks.bioagentbench.scorer_environment import child_environment
    from benchmarks.bioagentbench.worker import stage_owned_inputs, validate_identity

    if packet["task_id"] != "giab" or packet["image"] != IMAGE:
        raise ValueError("Qualification requires exact GIAB/image identity")
    taskset = packet["native_config"]["taskset"]
    if (
        taskset.get("task_id") != "giab"
        or taskset.get("staging") != "local"
        or taskset.get("include_reference") is not True
    ):
        raise ValueError("Qualification requires the unchanged local GIAB task selection")
    native_runtime = packet["native_config"]["harness"]["runtime"]
    expected_runtime = {
        "type": "docker",
        "image": IMAGE,
        "workdir": "/app",
        "cpu": 2,
        "memory": 8,
        "gpu": None,
        "disk": None,
    }
    if any(expected_runtime.get(key) != value for key, value in native_runtime.items()):
        raise ValueError("Unqualified runtime override")
    # Bind this helper too: child identity already validates the production modules.
    helper = str(Path(__file__).resolve())
    if packet["selected_sources"].get(helper) != hashlib.sha256(Path(helper).read_bytes()).hexdigest():
        raise ValueError("Unbound qualification helper")
    output.mkdir(parents=True, exist_ok=False)
    config = packet["native_config"]
    observed = validate_identity(packet, config)
    (output / "parent-imports.json").write_text(json.dumps(observed, indent=2))
    config = stage_owned_inputs(packet, config, output)
    source = Path(config["taskset"]["data_root"]) / "giab"
    if time.time() >= deadline:
        raise TimeoutError("Qualification staging exhausted the original deadline")
    workspace = output / "workspace"
    workspace.mkdir()

    class Base:
        def __init__(self, name):
            self.name = name

    class Config(SimpleNamespace):
        def model_dump(self):
            return vars(self)

    cls = runtime_class(
        workspace=workspace,
        source_dir=source,
        receipt=output / "sandbox-lifecycle.json",
        image=IMAGE,
        ttl_s=600,
        runtime_types=(Base, SimpleNamespace, SimpleNamespace),
    )
    runtime = cls(
        Config(type="docker", image=IMAGE, workdir="/app", cpu=2, memory=8, gpu=None, disk=None),
        name="giab-isolation-" + secrets.token_hex(8),
    )
    with tempfile.TemporaryDirectory(prefix="giab-isolation-", dir="/tmp") as temporary:
        os.chmod(temporary, 0o700)
        socket = Path(temporary) / "runtime.sock"
        token = secrets.token_hex(32)
        server = RuntimeServer(
            runtime, socket=socket, token=token, deadline=deadline, receipts=output / "runtime-operations"
        )
        try:
            await server.start()
            await runtime.start()
            inspected = json.loads(await command(["docker", "inspect", runtime.info.id]))[0]
            mounts = check_mounts(inspected, workspace=workspace, source=source)
            # Store only relevant inspection fields, never environment/credential values.
            (output / "mounts.json").write_text(
                json.dumps(
                    {
                        "Id": inspected["Id"],
                        "Image": inspected["Image"],
                        "Mounts": mounts,
                        "HostConfig": {
                            k: inspected["HostConfig"].get(k)
                            for k in ("NetworkMode", "Memory", "NanoCpus", "CpuPeriod", "CpuQuota")
                        },
                    },
                    indent=2,
                )
            )
            child_packet = {
                **packet,
                "native_config": config,
                "runtime_socket": str(socket),
                "ipc_token": token,
                "deadline": deadline,
                "original_data_root": packet["native_config"]["taskset"]["data_root"],
            }
            packet_path = Path(temporary) / "packet.json"
            packet_path.write_text(json.dumps(child_packet))
            packet_path.chmod(0o600)
            await command(
                [
                    packet["interpreter"]["path"],
                    "-m",
                    "benchmarks.bioagentbench.qualify_giab_isolation",
                    "--child",
                    "--packet",
                    str(packet_path),
                    "--output",
                    str(output),
                ],
                env=child_environment(packet, home=output, source=Path(__file__).resolve().parents[2]),
                timeout=max(0.001, min(180, deadline - time.time())),
            )
            files = json.loads(Path(packet["asset_manifest"]).read_text())["tasks"]["giab"]["files"]
            public = [str(source / name) for name in files if name.split("/")[0] in ("data", "reference")]
            forbidden = [
                str(source / "results"),
                str(Path(packet["native_config"]["taskset"]["data_root"]) / "giab/results"),
                packet["creator_root"],
                packet["giab_scorer_binding"]["root"],
                temporary,
            ]
            script = (
                "import os,json; p="
                + repr(public)
                + "; f="
                + repr(forbidden)
                + "; assert all(os.path.isfile(x) and os.access(x,os.R_OK) for x in p); assert all(not os.path.exists(x) for x in f); assert os.path.realpath('data')=="
                + repr(str(source / "data"))
                + "; assert os.path.realpath('reference')=="
                + repr(str(source / "reference"))
                + "; print(json.dumps({'public_readable':p,'forbidden_absent':f}))"
            )
            checked = await runtime.run(["python", "-c", script], {})
            if checked.exit_code:
                raise ValueError("Container public/truth visibility check failed")
            (output / "visibility.json").write_text(checked.stdout)
        finally:
            await server.close()
    (output / "qualification.json").write_text(
        json.dumps(
            {
                "passed": True,
                "provider_calls": 0,
                "scope": "Actual creator setup/Runtime IPC/mount visibility; no science, hap.py, judge or harness",
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    packet = json.loads(args.packet.read_text())
    asyncio.run(child(packet, args.output) if args.child else qualify(packet, args.output.absolute()))


if __name__ == "__main__":
    main()
