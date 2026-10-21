# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import pytest


def test_actual_source_and_asset_identity_before_dispatch(tmp_path):
    import hashlib
    import json
    import sys
    from pathlib import Path

    from benchmarks.bioagentbench import worker

    root = Path(__file__).resolve().parents[3]
    workspace = root.parent.parent
    metadata = (
        workspace
        / "outputs/broad-eval-campaign/harbor/agent-expansion/sources/56/bioagent-bench-bioagent-bench-6d098b6/src/task_metadata.json"
    )
    creator = root.parent / "bioagent-experiments-upstream"
    if not metadata.exists() or not creator.exists():
        pytest.skip("Pinned native source fixture unavailable")
    asset = tmp_path / "assets/cystic-fibrosis/data"
    asset.mkdir(parents=True)
    (asset / "input").write_text("public input")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "tasks": {
                    "cystic-fibrosis": {
                        "files": {"data/input": hashlib.sha256((asset / "input").read_bytes()).hexdigest()}
                    }
                }
            }
        )
    )
    files = [
        *list((root / "benchmarks/bioagentbench").glob("*.py")),
        *list((root / "nemo_gym/sandbox").rglob("*.py")),
        root / "nemo_gym/server_utils.py",
    ]
    packet = {
        "creator_root": str(creator),
        "selected_sources": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        "parent_interpreter": {
            "path": os.path.abspath(sys.executable),
            "sha256": hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
        },
        "asset_manifest": str(manifest),
        "asset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "task_id": "cystic-fibrosis",
    }
    config = {"taskset": {"metadata_path": str(metadata), "data_root": str(asset.parent.parent)}}
    native_python = root.parent / "bioagent-runtime/bin/python"
    if not native_python.exists():
        pytest.skip("Separate pinned creator interpreter unavailable")
    packet["interpreter"] = {
        "path": str(native_python.absolute()),
        "sha256": hashlib.sha256(native_python.resolve().read_bytes()).hexdigest(),
    }
    identity = worker.validate_identity(packet, config)
    import subprocess

    packet["native_config"] = config
    packet["digest"] = "offline-control"
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(str(Path(p).absolute()) for p in sys.path if p)
    child = subprocess.run(
        [sys.executable, "-m", "benchmarks.bioagentbench.worker", "--packet", str(packet_path), "--validate-only"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr
    child_identity = json.loads((tmp_path / "actual-imports.json").read_text())
    assert child_identity["prefix"] == sys.prefix
    assert Path(child_identity["creator_runtime"]["imports"]["bioagent_taskset.taskset"]["path"]).is_relative_to(
        tmp_path / "creator-source"
    )
    assert not (tmp_path / "creator-source/.pytest_cache").exists()
    assert (
        child_identity["imports"]["bioagent_worker"]["sha256"]
        == hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest()
    )
    assert not (tmp_path / "execution").exists()
    assert child_identity["creator_runtime"]["gym_imported"] is False
    assert "bioagent_taskset.taskset" not in identity["imports"]
    (asset / "input").write_text("changed")
    with pytest.raises(ValueError, match="asset packet differs"):
        worker.validate_identity(packet, config)
