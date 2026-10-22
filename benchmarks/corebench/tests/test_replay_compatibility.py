import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import harbor
import pytest

from benchmarks.corebench.replay_compatibility import stage_symlink_preserving_harbor


def test_real_single_step_preserves_links_without_reading_host_target(tmp_path):
    root = Path(harbor.__file__).parent
    receipt = stage_symlink_preserving_harbor(root, tmp_path / "overlay")
    spec = importlib.util.spec_from_file_location("patched_regrade", tmp_path / "overlay/harbor/trial/regrade.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    (source / "agent").mkdir(parents=True)
    (source / "artifacts/workspace").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "must-not-copy").write_text("host-only")
    (source / "artifacts/manifest.json").write_text("[]")
    workspace = source / "artifacts/workspace"
    (workspace / "report.json").write_text('{"answer": 1}')
    links = {"absolute": "/workspace/capsule/data", "dangling": "missing", "host": str(outside), "cycle": "."}
    for name, text in links.items():
        (workspace / name).symlink_to(text)
    target = tmp_path / "target"
    fake = SimpleNamespace(
        _source_paths=SimpleNamespace(
            agent_dir=source / "agent", artifacts_dir=source / "artifacts", trial_dir=source
        ),
        paths=SimpleNamespace(agent_dir=target / "agent", artifacts_dir=target / "artifacts"),
        task=SimpleNamespace(config=SimpleNamespace(artifacts=["/workspace"])),
        config=SimpleNamespace(artifacts=[]),
        _materialize_empty_artifacts=lambda **kwargs: None,
    )
    fake._replay_recorded_outputs = lambda **kwargs: module.RegradeTrial._replay_recorded_outputs(fake, **kwargs)
    module.RegradeTrial._seed_from_source(fake)
    for name, text in links.items():
        p = target / "artifacts/workspace" / name
        assert p.is_symlink() and os.readlink(p) == text
        assert os.readlink(workspace / name) == text
    assert (target / "artifacts/workspace/report.json").read_bytes() == (workspace / "report.json").read_bytes()
    assert receipt["installed_dependency_modified"] is False
    with pytest.raises(ValueError, match="already exists"):
        stage_symlink_preserving_harbor(root, tmp_path / "overlay")


def test_unreviewed_dependency_rejected_before_overlay(tmp_path):
    source = tmp_path / "source"
    (source / "trial").mkdir(parents=True)
    (source / "trial/regrade.py").write_text("changed")
    with pytest.raises(ValueError, match="Unreviewed"):
        stage_symlink_preserving_harbor(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_real_docker_tar_upload_preserves_links(tmp_path):
    import io
    import tarfile

    from harbor.environments.docker.docker_unix import UnixOps

    source = tmp_path / "workspace"
    source.mkdir()
    outside = tmp_path / "host-only"
    outside.mkdir()
    (outside / "secret").write_text("not-an-artifact")
    links = {"absolute": "/workspace/capsule/data", "dangling": "missing", "host": str(outside), "cycle": "."}
    for name, text in links.items():
        (source / name).symlink_to(text)
    (source / "report.json").write_text("saved answer")
    platform = object.__new__(UnixOps)
    payload = platform._tar_dir_contents(source)
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        members = {m.name: m for m in archive.getmembers()}
        assert set(members) == set(links) | {"report.json"}
        for name, text in links.items():
            assert members[name].issym() and members[name].linkname == text
        assert archive.extractfile(members["report.json"]).read() == b"saved answer"


@pytest.mark.asyncio
async def test_real_docker_upload_does_not_request_dereference(tmp_path):
    from harbor.environments.docker.docker_unix import UnixOps

    calls = []

    async def compose(args, **kwargs):
        calls.append(args)

    source = tmp_path / "workspace"
    source.mkdir()
    (source / "absolute").symlink_to("/workspace/data")
    await UnixOps(SimpleNamespace(_run_docker_compose_command=compose)).upload_dir(source, "/workspace")
    assert calls == [["cp", str(source) + "/.", "main:/workspace"]]
    assert os.readlink(source / "absolute") == "/workspace/data"
