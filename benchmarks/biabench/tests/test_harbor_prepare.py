# SPDX-License-Identifier: Apache-2.0
import pytest

from benchmarks.biabench import harbor_prepare


def test_historical_default_and_explicit_image(monkeypatch):
    monkeypatch.delenv("BIABENCH_SOLVER_IMAGE", raising=False)
    assert harbor_prepare.solver_image() == harbor_prepare.SOLVER_IMAGE
    image = "sha256:" + "a" * 64
    monkeypatch.setenv("BIABENCH_SOLVER_IMAGE", image)
    assert harbor_prepare.solver_image() == image


@pytest.mark.parametrize("image", ["latest", "sha256:123", "", "sha256:" + "A" * 64])
def test_invalid_image_refuses_before_staging(monkeypatch, image):
    monkeypatch.setenv("BIABENCH_SOLVER_IMAGE", image)
    monkeypatch.setattr(harbor_prepare, "stage", lambda *args: pytest.fail("staging must not run"))
    with pytest.raises(ValueError, match="immutable full"):
        harbor_prepare.prepare()


def test_override_binds_both_task_and_dockerfile_and_refuses_stale_cache(tmp_path, monkeypatch):
    import json

    source, task, assets = (tmp_path / n for n in ("source", "task", "assets"))
    for key, path in (("BIABENCH_SOURCE", source), ("BIABENCH_TASK", task), ("BIABENCH_DATA", assets)):
        monkeypatch.setenv(key, str(path))
    (assets / "solver").mkdir(parents=True)
    (assets / "solver/instruction.md").write_text("unaltered instruction")
    (assets / "benchmark.jsonl").write_text(json.dumps({"task_id": harbor_prepare.TASK}))
    monkeypatch.setattr(harbor_prepare, "stage", lambda *args: None)
    (tmp_path / "harbor_prepare.py").write_text("control source")
    (tmp_path / "frozen_sources.json").write_text(json.dumps({"creator_files": {}, "task_files": {}}))
    monkeypatch.setattr(harbor_prepare, "__file__", str(tmp_path / "harbor_prepare.py"))
    image = "sha256:" + "a" * 64
    monkeypatch.setenv("BIABENCH_SOLVER_IMAGE", image)
    harbor_prepare.prepare()
    root = tmp_path / "data/tasks" / harbor_prepare.TASK
    before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert f'docker_image = "{image}"' in (root / "task.toml").read_text()
    assert (root / "environment/Dockerfile").read_text() == f"FROM {image}\n"
    assert (root / "instruction.md").read_text() == "unaltered instruction"
    assert harbor_prepare.is_prepared_data_current(tmp_path / "data/benchmark.jsonl")
    for relative in ("task.toml", "environment/Dockerfile", "instruction.md"):
        path = root / relative
        original = path.read_bytes()
        path.write_bytes(original + b"tamper")
        assert not harbor_prepare.is_prepared_data_current(tmp_path / "data/benchmark.jsonl")
        path.write_bytes(original)
    monkeypatch.setenv("BIABENCH_SOLVER_IMAGE", "sha256:" + "b" * 64)
    assert not harbor_prepare.is_prepared_data_current(tmp_path / "data/benchmark.jsonl")
    with pytest.raises(Exception, match="Stale BIABench"):
        harbor_prepare.prepare()
    assert before == {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
