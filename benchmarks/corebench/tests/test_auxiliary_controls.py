# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import subprocess

import pytest

from benchmarks.corebench.auxiliary_controls import probe_contract


def example(executable):
    checksum = hashlib.sha256(b"original Dockerfile").hexdigest()
    row = {
        "id": "capsule-test",
        "published_image": "repo@sha256:" + "a" * 64,
        "medium_file_manifest": [{"path": "environment/Dockerfile", "sha256": checksum}],
    }
    specs = [
        {"executable": executable, "version_arg": "--version", "source_files": {"environment/Dockerfile": checksum}}
    ]
    return row, specs


def test_actual_command_records_version_hash_and_missing(tmp_path):
    executable = tmp_path / "scientific-tool"
    executable.write_text("#!/bin/sh\nprintf 'original-engine-2.1.1\\n'\n")
    executable.chmod(0o755)
    row, specs = example(str(executable))
    missing = copy.deepcopy(specs[0])
    missing["executable"] = str(tmp_path / "missing")
    command, binding = probe_contract(row, specs + [missing])
    result = subprocess.run(["/bin/sh", "-c", command], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert "original-engine-2.1.1" in result.stdout
    assert hashlib.sha256(executable.read_bytes()).hexdigest() in result.stdout
    assert "version_exit=0" in result.stdout
    assert "availability=missing-or-not-executable" in result.stdout
    assert binding["command_sha256"] == hashlib.sha256(command.encode()).hexdigest()


def test_failed_version_is_observation_not_readiness(tmp_path):
    executable = tmp_path / "broken-tool"
    executable.write_text("#!/bin/sh\necho missing-library >&2\nexit 7\n")
    executable.chmod(0o755)
    command, _ = probe_contract(*example(str(executable)))
    result = subprocess.run(["/bin/sh", "-c", command], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert "missing-library" in result.stdout and "version_exit=7" in result.stdout


@pytest.mark.parametrize(
    "field,value",
    [
        ("executable", "x; touch /tmp/bad"),
        ("executable", "/tmp/../bin/tool"),
        ("version_arg", "--install"),
        ("source_files", {}),
        ("source_files", {"unknown": "a" * 64}),
        ("source_files", {"environment/Dockerfile": "b" * 64}),
    ],
)
def test_reject_unbound_or_executable_injection(field, value):
    row, specs = example("phantomjs")
    specs[0][field] = value
    with pytest.raises(ValueError):
        probe_contract(row, specs)


def test_identity_changes_are_bound_and_inputs_unchanged():
    row, specs = example("phantomjs")
    original = copy.deepcopy((row, specs))
    _, binding = probe_contract(row, specs)
    row["published_image"] = "repo@sha256:" + "b" * 64
    assert probe_contract(row, specs)[1]["contract_sha256"] != binding["contract_sha256"]
    row = original[0]
    assert specs == original[1]
    with pytest.raises(ValueError):
        probe_contract(row, specs * 2)
    with pytest.raises(ValueError):
        probe_contract(row, [])


@pytest.mark.asyncio
async def test_observer_reuses_owned_probe_and_publishes_diagnostic_only(tmp_path, monkeypatch):
    import json

    from benchmarks.corebench import auxiliary_controls, image_controls

    calls = []

    async def probe(image, command, destination, **kwargs):
        calls.append((image, command, kwargs))
        destination.mkdir()
        return {"image": {"Id": "sha256:" + "c" * 64}, "stdout": "availability=missing-or-not-executable\n"}

    monkeypatch.setattr(image_controls, "probe_image", probe)
    row, specs = example("phantomjs")
    receipt = await auxiliary_controls.observe_auxiliaries(row, specs, tmp_path / "observation")
    assert calls[0][0] == row["published_image"]
    assert calls[0][2] == {"cpus": 1, "memory_mb": 1024}
    assert "no scientific workflow or readiness acceptance" in receipt["scope"]
    assert not (tmp_path / "observation" / "qualification.json").exists()
    assert json.loads((tmp_path / "observation" / "auxiliary-observation.json").read_text()) == receipt


@pytest.mark.asyncio
async def test_failed_probe_does_not_publish_receipt(tmp_path, monkeypatch):
    from benchmarks.corebench import auxiliary_controls, image_controls

    async def probe(*args, **kwargs):
        raise ValueError("cleanup unconfirmed")

    monkeypatch.setattr(image_controls, "probe_image", probe)
    with pytest.raises(ValueError, match="cleanup unconfirmed"):
        await auxiliary_controls.observe_auxiliaries(*example("phantomjs"), tmp_path / "observation")
    assert not (tmp_path / "observation").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["same", "package-drift", "empty", "cleanup-failure"])
async def test_creator_r_startup_requires_observed_parity(tmp_path, monkeypatch, variant):
    from benchmarks.corebench import auxiliary_controls, image_controls

    calls = []
    original = 'creator-r-startup-v1\n/usr/local/lib/R/site-library\nelectionsBR\t0.3.2\n'

    async def probe(image, command, destination, **kwargs):
        calls.append((image, command, kwargs))
        destination.mkdir(parents=True)
        if len(calls) == 2 and variant == "cleanup-failure":
            raise ValueError("cleanup unconfirmed")
        text = "" if variant == "empty" else original
        if len(calls) == 2 and variant == "package-drift":
            text = text.replace("0.3.2", "0.5.0")
        return {"image": {"Id": image}, "stdout": text}

    monkeypatch.setattr(image_controls, "probe_image", probe)
    row, _ = example("Rscript")
    dest = tmp_path / "r-startup"
    if variant == "same":
        receipt = await auxiliary_controls.qualify_creator_r_startup(row, "sha256:" + "c" * 64, dest)
        assert receipt["creator_r_startup_equal"]
        assert "project startup" in receipt["scope"]
        assert "--vanilla" not in calls[0][1]
        assert calls[0][0] == row["published_image"]
        assert calls[1][0] == "sha256:" + "c" * 64
    else:
        with pytest.raises(ValueError):
            await auxiliary_controls.qualify_creator_r_startup(row, "sha256:" + "c" * 64, dest)
        assert not (dest / "creator-r-startup.json").exists()
