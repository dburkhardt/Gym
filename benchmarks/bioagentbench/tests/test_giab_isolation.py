"""Offline controls; actual Docker qualification is a separate admission."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.bioagentbench.qualify_giab_isolation import IMAGE, check_mounts, setup_only


def inspect_fixture(tmp_path):
    workspace, source = tmp_path / "workspace", tmp_path / "giab"
    result = {
        "Image": IMAGE,
        "Mounts": [
            {"Type": "bind", "Source": str(workspace), "Destination": "/app"},
            *(
                {"Type": "bind", "Source": str(source / category), "Destination": str(source / category)}
                for category in ("data", "reference")
            ),
        ],
        "HostConfig": {"NetworkMode": "bridge", "Memory": 8 * 1024**3, "NanoCpus": 2_000_000_000},
    }
    return result, workspace, source


def test_exact_mount_tree(tmp_path):
    result, workspace, source = inspect_fixture(tmp_path)
    assert len(check_mounts(result, workspace=workspace, source=source)) == 3


@pytest.mark.parametrize("change", ["ancestor", "truth", "duplicate", "volume", "image", "memory", "network", "cpu"])
def test_mount_or_runtime_drift_refused(tmp_path, change):
    result, workspace, source = inspect_fixture(tmp_path)
    if change == "ancestor":
        result["Mounts"][1]["Source"] = str(source.parent)
    elif change == "truth":
        result["Mounts"].append({"Type": "bind", "Source": str(source / "results"), "Destination": "/hidden"})
    elif change == "duplicate":
        result["Mounts"][1] = copy.deepcopy(result["Mounts"][0])
    elif change == "volume":
        result["Mounts"][0]["Type"] = "volume"
    elif change == "image":
        result["Image"] = "sha256:" + "0" * 64
    else:
        result["HostConfig"][{"memory": "Memory", "network": "NetworkMode", "cpu": "NanoCpus"}[change]] = 0
    with pytest.raises(ValueError):
        check_mounts(result, workspace=workspace, source=source)


@pytest.mark.asyncio
async def test_actual_pinned_creator_setup_only(tmp_path):
    # Execute the exact source method, not an imitation; optional creator dependencies
    # are unnecessary for this method's setup contract.
    import ast
    import os

    if not os.environ.get("BIOAGENT_TEST_CREATOR_ROOT"):
        pytest.skip("Set BIOAGENT_TEST_CREATOR_ROOT to the pinned creator checkout")
    root = Path(os.environ["BIOAGENT_TEST_CREATOR_ROOT"])
    path = root / "tasksets/bioagent/bioagent_taskset/taskset.py"
    pins = json.loads((Path(__file__).parents[1] / "source-sha256.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pins["creator"][str(path.relative_to(root))]
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BioAgentTask")
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "setup")
    # Annotation names belong to the optional creator environment.
    for arg in method.args.args:
        arg.annotation = None
    method.returns = None
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    events = []

    class RecordingRuntime:
        async def run(self, argv, env):
            events.append((argv, env))
            return SimpleNamespace(exit_code=0)

    task = SimpleNamespace(
        data=SimpleNamespace(task_id="giab", staging="local", include_reference=True, source_dir=str(tmp_path))
    )

    async def setup(trace, runtime):
        await namespace["setup"](task, trace, runtime)

    task.setup = setup
    await setup_only(task, RecordingRuntime())
    assert events == [
        (["mkdir", "-p", "outputs", "results"], {}),
        (["ln", "-s", str(tmp_path / "data"), "data"], {}),
        (["ln", "-s", str(tmp_path / "reference"), "reference"], {}),
    ]
    task.data.staging = "download"
    with pytest.raises(ValueError):
        await setup_only(task, RecordingRuntime())


def test_trusted_child_checks_original_and_owned_truth(tmp_path):
    from benchmarks.bioagentbench.qualify_giab_isolation import trusted_truth

    original, owned = tmp_path / "original", tmp_path / "owned"
    digest = hashlib.sha256(b"truth").hexdigest()
    for root in (original, owned):
        (root / "giab/results").mkdir(parents=True)
        (root / "giab/results/truth.vcf").write_bytes(b"truth")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": {"giab": {"files": {"results/truth.vcf": digest}}}}))
    packet = {
        "asset_manifest": str(manifest),
        "asset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "original_data_root": str(original),
        "native_config": {"taskset": {"data_root": str(owned)}},
    }
    assert len(trusted_truth(packet)) == 2
    (original / "giab/results/truth.vcf").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        trusted_truth(packet)
    (original / "giab/results/truth.vcf").write_bytes(b"truth")
    (owned / "giab/results/truth.vcf").unlink()
    (owned / "giab/results/truth.vcf").symlink_to(original / "giab/results/truth.vcf")
    with pytest.raises(ValueError, match="regular"):
        trusted_truth(packet)


@pytest.fixture
def short_socket():
    import tempfile

    with tempfile.TemporaryDirectory(prefix="giab-test-", dir="/tmp") as directory:
        yield Path(directory) / "runtime.sock"


@pytest.mark.asyncio
async def test_real_runtime_channel_requires_start_handshake(tmp_path, short_socket):
    import time

    from benchmarks.bioagentbench.qualify_giab_isolation import setup_over_channel
    from benchmarks.bioagentbench.runtime_channel import RuntimeClient, RuntimeServer

    events = []

    class Backend:
        info = SimpleNamespace(id="fixture")

        async def start(self):
            events.append("start")

        async def run(self, argv, env):
            events.append(argv)
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

        async def teardown(self):
            events.append("teardown")

    backend = Backend()
    server = RuntimeServer(
        backend,
        socket=short_socket,
        token="fixture-token",
        deadline=time.time() + 20,
        receipts=tmp_path / "receipts",
    )
    await server.start()
    client = RuntimeClient(short_socket, "fixture-token", deadline=time.time() + 20)

    async def setup(trace, runtime):
        await runtime.run(["mkdir", "-p", "outputs", "results"], {})

    task = SimpleNamespace(data=SimpleNamespace(task_id="giab", staging="local", include_reference=True), setup=setup)
    try:
        await setup_over_channel(task, client)
        assert server.started and not server.failed
        assert client.ordinal == 2
        assert events == ["start", ["mkdir", "-p", "outputs", "results"]]
    finally:
        await server.close()
    assert events[-1] == "teardown"
    receipts = sorted((tmp_path / "receipts").glob("*.complete.json"))
    assert [json.loads(p.read_text())["method"] for p in receipts] == ["start", "run"]
