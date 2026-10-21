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
    assert parsed["artifacts"] == ["/workspace"]
    with pytest.raises(ValueError, match="Immutable"):
        runtime_config(original, "mutable-tag")
