# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest

from benchmarks.corebench.image_controls import dockerfile, qualification_digest, recipe_digest, source_digest
from benchmarks.corebench.prepare_frozen20 import stage, validate_qualification


def fixture(tmp):
    medium = tmp / "medium"
    refs = tmp / "refs"
    cid = "capsule-test"
    (medium / cid / "code").mkdir(parents=True)
    (medium / cid / "code/run.py").write_text("print(1)")
    (refs / cid / "solver-input").mkdir(parents=True)
    (refs / cid / "evaluator-only").mkdir()
    files = {
        "solver-input/instruction.txt": "original task",
        "evaluator-only/answers.json": '[{"value":1}]',
        "evaluator-only/evaluations.py": "creator scorer",
    }
    for name, value in files.items():
        (refs / cid / name).write_text(value)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    row = {
        "id": cid,
        "published_image": "registry.test/source@sha256:" + "a" * 64,
        "runtime_environment_variant": {},
        "status": "assets_pinned_runtime_unmeasured",
        "medium_file_manifest": [{"path": "code/run.py", "sha256": digest(medium / cid / "code/run.py")}],
        "instruction_sha256": digest(refs / cid / "solver-input/instruction.txt"),
        "reference_sha256": digest(refs / cid / "evaluator-only/answers.json"),
        "scorer_sha256": digest(refs / cid / "evaluator-only/evaluations.py"),
    }
    q = {
        "capsule": cid,
        "creator_image": row["published_image"],
        "final_image": "sha256:" + "b" * 64,
        "recipe_sha256": recipe_digest(row),
        "source_sha256": source_digest(row),
        "qualification_sha256": qualification_digest(row),
        "scientific_before": "python3 creator",
        "scientific_after": "python3 creator",
        "harness_python_ok": True,
        "tmux_ok": True,
        "asciinema_ok": True,
        "reference_isolation_ok": True,
        "cleanup_ok": True,
        "image_source_verified": True,
        "image_source_modes": {"code/run.py": 0o644},
        "resources": {"cpus": 2, "memory_mb": 4096, "storage_mb": 10240, "agent_timeout_sec": 3600},
    }
    manifest = tmp / "manifest.json"
    manifest.write_text(json.dumps({"rows": [row]}))
    return medium, refs, row, q, manifest


def test_sources_and_qualification_gate_publication(tmp_path):
    medium, refs, row, q, manifest = fixture(tmp_path)
    with pytest.raises(ValueError, match="No qualified"):
        stage(medium, refs, {}, tmp_path / "out", manifest=manifest)
    assert not (tmp_path / "out").exists()
    p = stage(medium, refs, {row["id"]: q}, tmp_path / "out", manifest=manifest)
    assert json.loads(p.read_text())["task_id"] == "corebench::capsule-test"
    task = p.parent / "tasks/capsule-test"
    assert not list((task / "environment").rglob("answers.json"))
    assert (task / "tests/item_diagnostics.py").is_file()
    assert not list((task / "environment").rglob("item_diagnostics.py"))
    assert (task / "tests/evaluations.py").read_bytes() == (
        refs / row["id"] / "evaluator-only/evaluations.py"
    ).read_bytes()
    assert "/opt/corebench-harness/bin/python3" in (task / "tests/test.sh").read_text()
    assert json.loads((p.parent / "preparation.json").read_text())["retained"] == 1


def test_changed_science_or_source_rejected(tmp_path):
    medium, refs, row, q, manifest = fixture(tmp_path)
    q["scientific_after"] = "different python"
    with pytest.raises(ValueError, match="interpreter"):
        validate_qualification(row, q)
    q["scientific_after"] = q["scientific_before"]
    (medium / row["id"] / "code/run.py").write_text("changed")
    with pytest.raises(ValueError, match="Changed medium"):
        stage(medium, refs, {row["id"]: q}, tmp_path / "out", manifest=manifest)
    assert not (tmp_path / "out").exists()


def test_cuda_oom_and_recipe_require_explicit_controls(tmp_path):
    _, _, row, q, _ = fixture(tmp_path)
    row["status"] = "requires_cuda_no_cpu_substitution"
    with pytest.raises(ValueError, match="CUDA"):
        validate_qualification(row, q)
    row["status"] = "measured_oom_requires_resource_review"
    with pytest.raises(ValueError, match="OOM"):
        validate_qualification(row, q)
    q["oom_resource_review"] = True
    with pytest.raises(ValueError, match="evidence-bound"):
        validate_qualification(row, q)
    row["status"] = "ready"
    q["recipe_sha256"] = "wrong"
    with pytest.raises(ValueError, match="recipe"):
        validate_qualification(row, q)


def test_recipe_never_replaces_scientific_python_path(tmp_path):
    _, _, row, _, _ = fixture(tmp_path)
    text = dockerfile(row)
    assert "ENV PATH" not in text
    assert "COPY python /opt/corebench-harness" in text
    assert "RUN python" not in text
    assert "/opt/corebench-harness/bin/python3 -m pip" in text


def test_unqualified_capsules_remain_visible(tmp_path):
    import shutil

    medium, refs, row, q, manifest = fixture(tmp_path)
    blocked = {**row, "id": "capsule-blocked", "status": "requires_cuda_no_cpu_substitution"}
    shutil.copytree(medium / row["id"], medium / blocked["id"])
    shutil.copytree(refs / row["id"], refs / blocked["id"])
    manifest.write_text(json.dumps({"rows": [row, blocked]}))
    p = stage(medium, refs, {row["id"]: q}, tmp_path / "out", manifest=manifest)
    receipt = json.loads((p.parent / "preparation.json").read_text())
    assert receipt["retained"] == 2 and receipt["published"] == 1
    assert receipt["blocked"] == [{"id": "capsule-blocked", "status": "requires_cuda_no_cpu_substitution"}]


@pytest.mark.parametrize("field", ["instruction_sha256", "reference_sha256", "scorer_sha256"])
def test_qualification_binds_task_and_reference(tmp_path, field):
    _, _, row, receipt, _ = fixture(tmp_path)
    row[field] = "f" * 64
    with pytest.raises(ValueError, match="binding missing or stale"):
        validate_qualification(row, receipt)


def test_qualification_binds_probe_and_refuses_historical_upgrade(tmp_path, monkeypatch):
    import benchmarks.corebench.image_controls as controls

    _, _, row, receipt, _ = fixture(tmp_path)
    historical = dict(receipt)
    historical.pop("qualification_sha256")
    with pytest.raises(ValueError, match="binding missing or stale"):
        validate_qualification(row, historical)
    monkeypatch.setattr(controls, "SCIENTIFIC_PROBE", controls.SCIENTIFIC_PROBE + "\necho changed\n")
    with pytest.raises(ValueError, match="binding missing or stale"):
        validate_qualification(row, receipt)


def test_r_probe_records_packages_and_library_paths():
    from benchmarks.corebench.image_controls import SCIENTIFIC_PROBE

    assert "installed.packages()" in SCIENTIFIC_PROBE
    assert ".libPaths()" in SCIENTIFIC_PROBE
    assert '"Package", "Version", "LibPath", "Built"' in SCIENTIFIC_PROBE


def test_qualification_binds_recovered_runtime(tmp_path, monkeypatch):
    import benchmarks.corebench.prepare as preparation

    _, _, row, receipt, _ = fixture(tmp_path)
    monkeypatch.setattr(preparation, "RUNTIME_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="binding missing or stale"):
        validate_qualification(row, receipt)


@pytest.mark.asyncio
@pytest.mark.parametrize("gpus", [0, 1])
async def test_maintained_qualification_producer(tmp_path, monkeypatch, gpus):
    import benchmarks.corebench.image_controls as controls
    import benchmarks.corebench.prepare as preparation

    medium, refs, row, _, _ = fixture(tmp_path)
    if gpus:
        row["status"] = "requires_cuda_no_cpu_substitution"
    runtime = tmp_path / "runtime.tar"
    runtime.write_bytes(b"fixture runtime")
    monkeypatch.setattr(preparation, "RUNTIME_SHA256", hashlib.sha256(runtime.read_bytes()).hexdigest())
    calls = []

    async def probe(image, command, destination, **kwargs):
        calls.append((image, command, kwargs))
        return {
            "image": {"Id": "sha256:" + "b" * 64},
            "stdout": "harness-python-ok reference-isolation-ok"
            if command == controls.HARNESS_PROBE
            else ("image-source-ok" if destination.name == "image-source" else "R package inventory"),
        }

    monkeypatch.setattr(controls, "probe_image", probe)
    receipt = await controls.qualify_images(
        row,
        medium,
        refs,
        runtime,
        "sha256:" + "b" * 64,
        tmp_path / "qualified",
        resources={
            "cpus": 2,
            "memory_mb": 4096,
            "storage_mb": 10240,
            "agent_timeout_sec": 3600,
            **({"gpus": gpus} if gpus else {}),
        },
    )
    validate_qualification(row, receipt)
    assert len(calls) == 4
    assert all(call[2].get("gpus", 0) == gpus for call in calls)
    assert calls[0][1] == calls[1][1] == controls.SCIENTIFIC_PROBE
    assert json.loads((tmp_path / "qualified/qualification.json").read_text()) == receipt


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["source", "identity"])
async def test_producer_refuses_wrong_image_source_or_identity(tmp_path, monkeypatch, failure):
    import benchmarks.corebench.image_controls as controls
    import benchmarks.corebench.prepare as preparation

    medium, refs, row, _, _ = fixture(tmp_path)
    runtime = tmp_path / "runtime.tar"
    runtime.write_bytes(b"fixture runtime")
    monkeypatch.setattr(preparation, "RUNTIME_SHA256", hashlib.sha256(runtime.read_bytes()).hexdigest())

    async def probe(image, command, destination, **kwargs):
        if destination.name == "image-source" and failure == "source":
            raise ValueError("image source inventory differs")
        return {
            "image": {
                "Id": "sha256:" + ("c" if destination.name == "harness" and failure == "identity" else "b") * 64
            },
            "stdout": "harness-python-ok reference-isolation-ok"
            if command == controls.HARNESS_PROBE
            else ("image-source-ok" if destination.name == "image-source" else "R inventory"),
        }

    monkeypatch.setattr(controls, "probe_image", probe)
    with pytest.raises(ValueError, match="inventory differs|identity changed"):
        await controls.qualify_images(
            row,
            medium,
            refs,
            runtime,
            "sha256:" + "b" * 64,
            tmp_path / "out",
            resources={"cpus": 2, "memory_mb": 4096, "storage_mb": 10240, "agent_timeout_sec": 3600},
        )
    assert not (tmp_path / "out/qualification.json").exists()


@pytest.mark.parametrize("mutation", [None, "bytes", "mode", "extra"])
def test_actual_image_source_check(tmp_path, mutation):
    import subprocess
    import sys

    from benchmarks.corebench.image_controls import image_source_probe

    medium, _, row, _, _ = fixture(tmp_path)
    root = medium / row["id"]
    file = root / "code/run.py"
    modes = {"code/run.py": file.stat().st_mode & 0o777}
    command = image_source_probe(row, modes, root)
    if mutation == "bytes":
        file.write_text("changed")
    elif mutation == "mode":
        file.chmod(0o700)
    elif mutation == "extra":
        (root / "unexpected").write_text("extra")
    result = subprocess.run([sys.executable, "-O", "-c", command], capture_output=True, text=True)
    assert (result.returncode == 0) is (mutation is None)
    assert ("image-source-ok" in result.stdout) is (mutation is None)


def test_context_excludes_staging_metadata_preserves_creator_lock_filename(tmp_path, monkeypatch):
    import tarfile

    import benchmarks.corebench.prepare as preparation
    from benchmarks.corebench.image_controls import stage_context

    medium, refs, row, _, _ = fixture(tmp_path)
    original_lock = medium / row["id"] / "code/.legitimate.stage.lock"
    original_lock.write_text("creator data, not a staging artifact")
    row["medium_file_manifest"].append(
        {"path": "code/.legitimate.stage.lock", "sha256": hashlib.sha256(original_lock.read_bytes()).hexdigest()}
    )
    runtime = tmp_path / "runtime"
    (runtime / "python/bin").mkdir(parents=True)
    (runtime / "python/bin/python3").write_text("fixture runtime")
    archive = tmp_path / "runtime.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(runtime / "python", arcname="python")
    monkeypatch.setattr(preparation, "RUNTIME_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    context = stage_context(row, medium, refs, archive, tmp_path / "context")
    actual = {str(p.relative_to(context / "capsule")) for p in (context / "capsule").rglob("*") if p.is_file()}
    assert actual == {f["path"] for f in row["medium_file_manifest"]}
    assert (context / "capsule/code/.legitimate.stage.lock").read_text() == original_lock.read_text()
    assert {str(p.relative_to(context)) for p in (context / "python").rglob("*") if p.is_file()} == {
        "python/bin/python3"
    }
    assert (tmp_path / ".context.stage.lock").exists()
    with pytest.raises(ValueError, match="existing"):
        stage_context(row, medium, refs, archive, context)


@pytest.mark.asyncio
async def test_large_source_probe_staged_with_short_argv(tmp_path, monkeypatch):
    import contextlib
    import subprocess
    import sys
    from types import SimpleNamespace

    import benchmarks.corebench.image_controls as controls
    import nemo_gym.sandbox as sandbox_api
    import nemo_gym.sandbox.lifecycle as lifecycle

    root = tmp_path / "capsule"
    root.mkdir()
    manifest, modes = [], {}
    for index in range(3686):
        name = f"scientific-file-{index:05d}.txt"
        path = root / name
        path.write_text(str(index))
        manifest.append({"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        modes[name] = path.stat().st_mode & 0o777
    script = controls.image_source_probe({"medium_file_manifest": manifest}, modes, root)
    assert len(script.encode()) > 131072
    uploaded = []
    executed = []

    class Sandbox:
        def __init__(self, *args):
            pass

        async def upload(self, local_path, remote_path):
            assert remote_path == "/tmp/corebench-image-probe.py"
            assert local_path.read_text() == script
            uploaded.append(local_path)

        async def exec(self, command, timeout_s):
            assert len(command.encode()) < 256
            assert timeout_s == 120
            executed.append(command)
            result = subprocess.run([sys.executable, "-I", str(uploaded[-1])], capture_output=True, text=True)
            return SimpleNamespace(
                stdout=result.stdout, stderr=result.stderr, return_code=result.returncode, error_type=None
            )

    @contextlib.asynccontextmanager
    async def owned(sandbox, receipt):
        try:
            yield sandbox
        finally:
            receipt.write_text(json.dumps({"cleanup_confirmed": True}))

    monkeypatch.setattr(sandbox_api, "AsyncSandbox", Sandbox)
    monkeypatch.setattr(lifecycle, "_owned_sandbox", owned)
    monkeypatch.setattr(controls, "inspect_local_image", lambda image: {"Id": image})
    command = "/opt/corebench-harness/bin/python3 -I /tmp/corebench-image-probe.py"
    await controls.probe_image("sha256:" + "b" * 64, command, tmp_path / "pass", script=script)
    receipt = json.loads((tmp_path / "pass/probe.json").read_text())
    assert receipt["script_sha256"] == hashlib.sha256(script.encode()).hexdigest()
    assert receipt["command_sha256"] == hashlib.sha256(command.encode()).hexdigest()
    (root / manifest[-1]["path"]).write_text("tampered last file")
    with pytest.raises(ValueError, match="Image probe failed"):
        await controls.probe_image("sha256:" + "b" * 64, command, tmp_path / "fail", script=script)
    assert len(executed) == 2


@pytest.mark.parametrize("override", [None, {"memory_mb": 8192}])
def test_execution_memory_preserves_qualification(tmp_path, override):
    import copy
    import tomllib

    medium, refs, row, q, manifest = fixture(tmp_path)
    original = copy.deepcopy(q)
    options = {} if override is None else {row["id"]: override}
    stage(medium, refs, {row["id"]: q}, tmp_path / "out", manifest=manifest, execution_resources=options)
    task = tomllib.loads((tmp_path / "out/tasks" / row["id"] / "task.toml").read_text())
    assert task["environment"]["memory_mb"] == (4096 if override is None else 8192)
    assert task["agent"]["timeout_sec"] == 3600
    assert task["verifier"]["timeout_sec"] == 120
    receipt = json.loads((tmp_path / "out/preparation.json").read_text())
    assert receipt["qualification_receipts"][row["id"]] == original == q
    assert receipt["execution_resources"][row["id"]]["memory_mb"] == task["environment"]["memory_mb"]


@pytest.mark.parametrize(
    "options",
    [
        {"unknown": {"memory_mb": 8192}},
        {"capsule-test": {"memory_mb": 2048}},
        {"capsule-test": {"memory_mb": True}},
        {"capsule-test": {"memory_mb": "8192"}},
        {"capsule-test": {"cpus": 4}},
        {"capsule-test": None},
    ],
)
def test_execution_resources_reject_invalid_before_publication(tmp_path, options):
    medium, refs, row, q, manifest = fixture(tmp_path)
    with pytest.raises(ValueError):
        stage(medium, refs, {row["id"]: q}, tmp_path / "out", manifest=manifest, execution_resources=options)
    assert not (tmp_path / "out").exists()


def gpu_evidence(tmp_path, row, q):
    from benchmarks.corebench.placement_evidence import REQUIRED

    root = tmp_path / "gpu-control"
    root.mkdir()
    records = {
        "admission.json": {
            "image": q["final_image"],
            "cpus": 2,
            "memory_mib": 8192,
            "gpu": 1,
            "scientific_timeout_s": 3600,
            "no_model_calls": True,
        },
        "container.json": {
            "Image": q["final_image"],
            "Name": "/owned",
            "HostConfig": {
                "NanoCpus": 2 * 10**9,
                "Memory": 8192 * 1024**2,
                "NetworkMode": "none",
                "DeviceRequests": [{"Count": 1, "Capabilities": [["gpu"]], "DeviceIDs": None}],
            },
        },
        "manifest.json": {f["path"]: f["sha256"] for f in row["medium_file_manifest"]},
        "before-source.json": {"files": len(row["medium_file_manifest"]), "changed": []},
        "after-source.json": {"files": len(row["medium_file_manifest"]), "changed": []},
        "execution.json": {"return_code": 0, "error_type": None},
        "independent-cleanup.json": {"sandbox_id": "owned", "absent": True},
    }
    for name, value in records.items():
        (root / name).write_text(json.dumps(value))
    (root / "stdout.txt").write_text("fresh scientific output")
    return {"root": str(root), "sha256": {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in REQUIRED}}


def test_gpu_execution_retains_cpu_qualification(tmp_path):
    import copy
    import tomllib

    medium, refs, row, q, manifest = fixture(tmp_path)
    original = copy.deepcopy(q)
    evidence = gpu_evidence(tmp_path, row, q)
    stage(
        medium,
        refs,
        {row["id"]: q},
        tmp_path / "out",
        manifest=manifest,
        execution_resources={row["id"]: {"memory_mb": 8192, "gpus": 1}},
        placement_evidence={row["id"]: evidence},
    )
    task = tomllib.loads((tmp_path / "out/tasks" / row["id"] / "task.toml").read_text())
    assert task["environment"]["gpus"] == 1 and task["environment"]["memory_mb"] == 8192
    assert q == original and not q["resources"].get("gpus")
    receipt = json.loads((tmp_path / "out/preparation.json").read_text())
    assert receipt["qualification_receipts"][row["id"]] == original
    assert receipt["execution_placement_reviews"][row["id"]]["sha256"] == evidence["sha256"]


@pytest.mark.parametrize("mutation", ["missing", "bytes", "image", "sources", "gpu", "timeout", "failed", "cleanup"])
def test_gpu_execution_rejects_unqualified_placement(tmp_path, mutation):
    from pathlib import Path

    medium, refs, row, q, manifest = fixture(tmp_path)
    evidence = gpu_evidence(tmp_path, row, q)
    root = Path(evidence["root"])
    if mutation == "missing":
        evidence = None
    elif mutation == "bytes":
        (root / "stdout.txt").write_text("changed")
    else:
        name, field, value = {
            "image": ("admission.json", "image", "sha256:" + "c" * 64),
            "sources": ("manifest.json", "unexpected", "a" * 64),
            "gpu": (
                "container.json",
                "HostConfig",
                {"NanoCpus": 2 * 10**9, "Memory": 8192 * 1024**2, "NetworkMode": "none", "DeviceRequests": []},
            ),
            "timeout": ("admission.json", "scientific_timeout_s", 7200),
            "failed": ("execution.json", "return_code", 1),
            "cleanup": ("independent-cleanup.json", "sandbox_id", "unrelated"),
        }[mutation]
        record = json.loads((root / name).read_text())
        record[field] = value
        (root / name).write_text(json.dumps(record))
        evidence["sha256"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        stage(
            medium,
            refs,
            {row["id"]: q},
            tmp_path / "out",
            manifest=manifest,
            execution_resources={row["id"]: {"memory_mb": 8192, "gpus": 1}},
            placement_evidence={row["id"]: evidence},
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("with_evidence", [False, True])
def test_gpu_qualification_still_requires_workflow_placement(tmp_path, with_evidence):
    medium, refs, row, q, manifest = fixture(tmp_path)
    q["resources"].update(memory_mb=8192, gpus=1)
    evidence = gpu_evidence(tmp_path, row, q)
    kwargs = {"placement_evidence": {row["id"]: evidence}} if with_evidence else {}
    if not with_evidence:
        with pytest.raises(ValueError, match="separate placement evidence"):
            stage(medium, refs, {row["id"]: q}, tmp_path / "out", manifest=manifest, **kwargs)
        assert not (tmp_path / "out").exists()
    else:
        stage(medium, refs, {row["id"]: q}, tmp_path / "out", manifest=manifest, **kwargs)
        receipt = json.loads((tmp_path / "out/preparation.json").read_text())
        assert row["id"] in receipt["execution_placement_reviews"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,gpus",
    [
        ("requires_cuda_no_cpu_substitution", 0),
        ("requires_cuda_no_cpu_substitution", True),
        ("measured_oom_requires_resource_review", 1),
    ],
)
async def test_image_producer_rejects_ineligible_resources_before_probe(tmp_path, monkeypatch, status, gpus):
    import benchmarks.corebench.image_controls as controls
    import benchmarks.corebench.prepare as preparation

    medium, refs, row, _, _ = fixture(tmp_path)
    row["status"] = status
    runtime = tmp_path / "runtime.tar"
    runtime.write_bytes(b"runtime")
    monkeypatch.setattr(preparation, "RUNTIME_SHA256", hashlib.sha256(runtime.read_bytes()).hexdigest())

    async def forbidden(*args, **kwargs):
        pytest.fail("ineligible resource configuration reached probe")

    monkeypatch.setattr(controls, "probe_image", forbidden)
    with pytest.raises(ValueError, match="GPU|OOM"):
        await controls.qualify_images(
            row,
            medium,
            refs,
            runtime,
            "sha256:" + "b" * 64,
            tmp_path / "out",
            resources={"cpus": 2, "memory_mb": 8192, "storage_mb": 10240, "agent_timeout_sec": 3600, "gpus": gpus},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("gpus", [0, 1])
async def test_image_probe_binds_actual_sandbox_gpu_request(tmp_path, monkeypatch, gpus):
    import contextlib
    from types import SimpleNamespace

    import benchmarks.corebench.image_controls as controls
    import nemo_gym.sandbox as api
    import nemo_gym.sandbox.lifecycle as lifecycle

    specs = []

    class Sandbox:
        def __init__(self, config, spec):
            specs.append(spec)
            assert config["docker"]["create"]["network"] == "none"

        async def exec(self, command, timeout_s):
            return SimpleNamespace(stdout="ok", stderr="", return_code=0, error_type=None)

    @contextlib.asynccontextmanager
    async def owned(sandbox, receipt):
        yield sandbox
        receipt.write_text(json.dumps({"cleanup_confirmed": True}))

    monkeypatch.setattr(api, "AsyncSandbox", Sandbox)
    monkeypatch.setattr(lifecycle, "_owned_sandbox", owned)
    monkeypatch.setattr(controls, "inspect_local_image", lambda image: {"Id": image})
    await controls.probe_image("sha256:" + "b" * 64, "true", tmp_path / "probe", gpus=gpus)
    assert specs[0].resources.gpu == gpus
    assert specs[0].provider_options == ({"run_args": ["--runtime=nvidia"]} if gpus else {})
