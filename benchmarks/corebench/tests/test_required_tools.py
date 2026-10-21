# SPDX-License-Identifier: Apache-2.0
import os
import subprocess

import pytest

from benchmarks.corebench.auxiliary_controls import required_tool_contract, validate_required_tools


def contract():
    row = {
        "id": "capsule-test",
        "published_image": "immutable",
        "medium_file_manifest": [{"path": "Dockerfile", "sha256": "a" * 64}],
    }
    selections = [{"executable": "selectedtool", "version_arg": "--version", "source_files": {"Dockerfile": "a" * 64}}]
    return row, selections


def run_probe(tmp_path, *, status=0, missing=False):
    row, selections = contract()
    command, _ = required_tool_contract(row, selections)
    if not missing:
        tool = tmp_path / "selectedtool"
        tool.write_text(f"#!/bin/sh\necho version1\nexit {status}\n")
        tool.chmod(0o755)
    ldd = tmp_path / "ldd"
    ldd.write_text('#!/bin/sh\nprintf "\\tlibfixture => /bin/sh (0x12345)\\n"\n')
    ldd.chmod(0o755)
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        env=dict(os.environ, PATH=str(tmp_path) + ":" + os.environ["PATH"]),
        text=True,
        capture_output=True,
    )
    return result, selections


def test_generated_probe_actual_shell_and_required_success(tmp_path):
    result, selections = run_probe(tmp_path)
    assert result.returncode == 0, result.stderr
    normalized = validate_required_tools(result.stdout, selections)
    assert "/bin/sh" in normalized and "(ASLR)" in normalized


@pytest.mark.parametrize("missing,status", [(True, 0), (False, 7)])
def test_actual_missing_or_nonzero_refused(tmp_path, missing, status):
    result, selections = run_probe(tmp_path, missing=missing, status=status)
    with pytest.raises(ValueError):
        validate_required_tools(result.stdout, selections)


@pytest.mark.parametrize(
    "old,new",
    [
        ("version_exit=0", "version_exit=7"),
        ("ldd_exit=0", "ldd_exit=1"),
        ("resolved=/", "resolved=relative"),
        ("version1", "not found"),
    ],
)
def test_bad_observations_refused(tmp_path, old, new):
    result, selections = run_probe(tmp_path)
    with pytest.raises(ValueError):
        validate_required_tools(result.stdout.replace(old, new), selections)


def test_parity_preserves_versions_paths_hashes(tmp_path):
    result, selections = run_probe(tmp_path)
    before = validate_required_tools(result.stdout, selections)
    for old, new in [("version1", "version2"), ("/bin/sh", "/bin/bash"), ("resolved=/", "resolved=/changed/")]:
        assert before != validate_required_tools(result.stdout.replace(old, new), selections)
    assert before == validate_required_tools(result.stdout.replace("0x12345", "0x98765"), selections)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [None, "version", "library", "path"])
async def test_producer_refuses_changed_closure(tmp_path, monkeypatch, change):
    from benchmarks.corebench import image_controls
    from benchmarks.corebench.auxiliary_controls import qualify_required_tools

    result, selections = run_probe(tmp_path)
    outputs = [result.stdout, result.stdout]
    if change == "version":
        outputs[1] = outputs[1].replace("version1", "version2")
    if change == "path":
        outputs[1] = outputs[1].replace("/bin/sh", "/bin/bash")
    if change == "library":
        lines = outputs[1].splitlines()
        lines[-1] = "0" * 64 + "  /bin/sh"
        outputs[1] = "\n".join(lines) + "\n"

    async def probe(image, command, destination, **kwargs):
        destination.mkdir(parents=True)
        return {"image": {"Id": image}, "stdout": outputs.pop(0)}

    monkeypatch.setattr(image_controls, "probe_image", probe)
    row, _ = contract()
    destination = tmp_path / "paired"
    if change:
        with pytest.raises(ValueError, match="state changed"):
            await qualify_required_tools(row, selections, "sha256:" + "b" * 64, destination)
        assert not (destination / "required-tools.json").exists()
    else:
        receipt = await qualify_required_tools(row, selections, "sha256:" + "b" * 64, destination)
        assert receipt["required_tools_equal"] and (destination / "required-tools.json").exists()
