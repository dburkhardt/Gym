# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from benchmarks.airsbench.regrade import regrade_managed, validate_inputs


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_untrusted_workspace_special_files_rejected_before_read(tmp_path, kind):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if kind == "symlink":
        (workspace / "submission.csv").symlink_to(tmp_path / "outside")
    else:
        os.mkfifo(workspace / "submission.csv")
    with pytest.raises(ValueError, match="symlink or special file"):
        validate_inputs("GraphRegressionZincMae", tmp_path / "source", workspace)


async def test_incomplete_capture_cannot_be_scored_as_missing_model_output(tmp_path):
    with pytest.raises(ValueError, match="capture evidence"):
        await regrade_managed(
            task_id="GraphRegressionZincMae",
            source=tmp_path / "source",
            workspace=tmp_path / "workspace",
            output=tmp_path / "result",
        )
    assert not (tmp_path / "result").exists()


def test_missing_reference_cache_is_not_model_failure(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "submission.csv").write_text("prediction\n1\n")
    with pytest.raises(ValueError, match="scorer asset"):
        validate_inputs("GraphRegressionZincMae", tmp_path / "source", workspace)


def test_native_overlay_preserves_budgets_and_declares_separate_scoring():
    import tomllib

    from benchmarks.airsbench.native_prepare import runtime_config

    original = """version = "1.0"
[agent]
timeout_sec = 3600
[verifier]
timeout_sec = 300
[environment]
docker_image = "mutable-old-tag"
cpus = 2
memory_mb = 3072
allow_internet = false
"""
    image = "sha256:" + "a" * 64
    parsed = tomllib.loads(runtime_config(original, image))
    assert parsed["agent"]["timeout_sec"] == 3600
    assert parsed["verifier"] == {"timeout_sec": 300, "environment_mode": "separate"}
    assert parsed["environment"]["docker_image"] == image
    assert parsed["environment"]["network_mode"] == "no-network"
    assert parsed["environment"]["memory_mb"] == 3072
    assert parsed["artifacts"] == [{"source": "/workspace", "exclude": ["data", "./data"]}]
    with pytest.raises(ValueError, match="Immutable"):
        runtime_config(original, "mutable-tag")


def test_regrade_identity_records_scientific_and_controller_versions():
    import hashlib
    import json
    from pathlib import Path

    from benchmarks.airsbench import regrade

    manifest_path = Path(regrade.__file__).with_name("frozen_sources.json")
    manifest = json.loads(manifest_path.read_text())
    task = manifest["tasks"][0]
    identity = regrade.scorer_identity(task)
    assert identity["creator_revision"] == manifest["creator_revision"]
    assert identity["scorer_assets_sha256"] == {
        name: digest for name, digest in task["files"].items() if name.startswith("tests/")
    }
    assert "tests/check.py" in identity["scorer_assets_sha256"]
    assert identity["regrade_source_sha256"] == hashlib.sha256(Path(regrade.__file__).read_bytes()).hexdigest()
    changed = dict(task, files=dict(task["files"], **{"tests/check.py": "changed"}))
    assert regrade.scorer_identity(changed) != identity


def test_interrupted_overlay_accepts_known_files_but_rejects_stale_inputs(tmp_path):
    from pathlib import Path

    from benchmarks.airsbench.native_prepare import validate_existing_overlay

    target = tmp_path / "task"
    (target / "tests").mkdir(parents=True)
    from nemo_gym.dataset_staging import stage_file

    source = tmp_path / "original.py"
    source.write_text("partial")
    stage_file(source, target / "tests/check.py")
    assert (target / "tests/.check.py.stage.lock").exists()
    expected = {Path("tests/check.py"), Path("task.toml")}
    validate_existing_overlay(target, expected)
    stage_file(source, target / "tests/check.py")
    validate_existing_overlay(target, expected)
    (target / "old_answer.txt").write_text("stale")
    with pytest.raises(ValueError, match="Unexpected prepared task file"):
        validate_existing_overlay(target, expected)
    (target / "old_answer.txt").unlink()
    (target / "tests/check.py").unlink()
    (target / "tests/check.py").symlink_to(tmp_path / "external")
    with pytest.raises(ValueError, match="symlink"):
        validate_existing_overlay(target, expected)


def test_overlay_rejects_symlinked_ancestor_before_publication(tmp_path):
    from pathlib import Path

    from benchmarks.airsbench.native_prepare import validate_existing_overlay

    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        validate_existing_overlay(alias / "not-created", {Path("task.toml")})
    assert not (outside / "not-created").exists()


@pytest.mark.parametrize("extra", ["file", "symlink", "fifo"])
def test_scorer_upload_tree_contains_only_manifest_assets(tmp_path, monkeypatch, extra):
    import hashlib
    import json

    from benchmarks.airsbench import regrade

    source = tmp_path / "source"
    (source / "tests").mkdir(parents=True)
    script = source / "tests/check.py"
    script.write_text("pass")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest = {
        "tasks": [{"name": "control", "files": {"tests/check.py": hashlib.sha256(script.read_bytes()).hexdigest()}}]
    }
    (tmp_path / "frozen_sources.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(regrade, "__file__", str(tmp_path / "regrade.py"))
    regrade.validate_inputs("control", source, workspace)
    additional = source / "tests/extra.py"
    if extra == "file":
        additional.write_text("unrecorded")
    elif extra == "symlink":
        additional.symlink_to(script)
    else:
        os.mkfifo(additional)
    with pytest.raises(ValueError, match="unrecorded|symlink or special"):
        regrade.validate_inputs("control", source, workspace)


def test_metric_direction_and_cross_target_aggregation_disabled():
    import json
    from pathlib import Path

    import yaml

    from benchmarks.airsbench.regrade import metric_metadata

    config = yaml.safe_load(Path("benchmarks/airsbench/config.yaml").read_text())
    assert config["disable_aggregation"] is True
    manifest = json.loads(Path("benchmarks/airsbench/frozen_sources.json").read_text())
    for task in manifest["tasks"]:
        metric = metric_metadata(task["name"])
        assert metric["direction"] == "minimize"
        assert metric["cross_target_aggregation"] is False
        assert metric["units"] and metric["normalization"] is None
