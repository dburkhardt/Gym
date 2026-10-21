# SPDX-License-Identifier: Apache-2.0
import hashlib
import io
import tarfile

import pytest

from benchmarks.corral.state_artifact import pack_state, unpack_state
from nemo_gym.state_bundle import pack_bundle, unpack_bundle


def test_nested_state_roundtrip_and_partial_archive_refusal(tmp_path):
    source = tmp_path / "source"
    nested = source / "state/executions/identity"
    nested.mkdir(parents=True)
    (nested / "commits.sqlite3").write_bytes(b"persistent native database")
    (nested / "snapshot.json").write_text('{"commit":"immutable"}')
    (nested / "empty").mkdir()
    nested.chmod(0o2770)
    (nested / "snapshot.json").chmod(0o640)
    (source / "solver-result.json").write_text('{"status":"submitted"}')
    archive = tmp_path / "state.tar"
    manifest = pack_state(source, archive)
    unpack_state(archive, manifest, tmp_path / "result")
    assert all(
        hashlib.sha256((tmp_path / "result" / name).read_bytes()).hexdigest() == entry["sha256"]
        for name, entry in manifest["entries"].items()
        if entry["kind"] == "file"
    )
    assert (tmp_path / "result/state/executions/identity/empty").is_dir()
    assert (tmp_path / "result/state/executions/identity").stat().st_mode & 0o7777 == 0o2770
    assert (tmp_path / "result/state/executions/identity/snapshot.json").stat().st_mode & 0o777 == 0o640
    # Reproduces the observed transfer defect: JSON survived, nested state did not.
    partial = tmp_path / "partial.tar"
    with tarfile.open(partial, "w") as out:
        out.add(source / "solver-result.json", arcname="solver-result.json")
    manifest["archive_sha256"] = hashlib.sha256(partial.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="file set"):
        unpack_state(partial, manifest, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_reject_links_and_traversal(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="link"):
        pack_bundle(source, tmp_path / "state.tar")
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as out:
        member = tarfile.TarInfo("../escape")
        member.size = 1
        out.addfile(member, io.BytesIO(b"x"))
    manifest = {
        "schema_version": 1,
        "entries": {
            "../escape": {"kind": "file", "size": 1, "mode": 0o644, "sha256": hashlib.sha256(b"x").hexdigest()}
        },
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="Unsafe"):
        unpack_bundle(archive, manifest, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_mutation_after_validation_never_publishes(tmp_path, monkeypatch):
    import nemo_gym.state_bundle as module

    source = tmp_path / "source"
    source.mkdir()
    (source / "data").write_bytes(b"old")
    archive = tmp_path / "bundle.tar"
    manifest = pack_bundle(source, archive)
    validate = module._validate

    def mutate(path, receipt):
        validate(path, receipt)
        with tarfile.open(path, "w") as out:
            member = tarfile.TarInfo("data")
            member.size = 3
            out.addfile(member, io.BytesIO(b"new"))

    monkeypatch.setattr(module, "_validate", mutate)
    with pytest.raises(ValueError, match="checksum"):
        unpack_bundle(archive, manifest, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_actual_archive_member_limit_checked_during_iteration(tmp_path, monkeypatch):
    import nemo_gym.state_bundle as module

    archive = tmp_path / "bundle.tar"
    with tarfile.open(archive, "w") as out:
        for name in ("a", "b", "c"):
            out.addfile(tarfile.TarInfo(name), io.BytesIO(b""))
    manifest = {"schema_version": 1, "entries": {}, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    monkeypatch.setattr(module, "MAX_MEMBERS", 2)
    with pytest.raises(ValueError, match="Too many archive"):
        unpack_bundle(archive, manifest, tmp_path / "result")
    assert not (tmp_path / "result").exists()
