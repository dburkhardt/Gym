# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit permission-policy controls; no archived ownership restoration."""

import hashlib
import os
import tarfile

import pytest

from nemo_gym.state_bundle import DIRECTORY_SETGID_POLICY, pack_bundle, unpack_bundle


def tree(tmp_path):
    source = tmp_path / "source"
    directory = source / "execution"
    directory.mkdir(parents=True)
    (directory / "database").write_bytes(b"immutable sqlite stand-in")
    (directory / "database").chmod(0o640)
    directory.chmod(0o2770)
    return source


def test_legacy_default_and_new_explicit_policy(tmp_path):
    source = tree(tmp_path)
    old = pack_bundle(source, tmp_path / "old.tar")
    assert "permission_policy" not in old
    assert old["entries"]["execution"]["mode"] == 0o770
    unpack_bundle(tmp_path / "old.tar", old, tmp_path / "old")
    assert (tmp_path / "old/execution").stat().st_mode & 0o7777 == 0o770
    new = pack_bundle(source, tmp_path / "new.tar", preserve_directory_setgid=True)
    assert new["permission_policy"] == DIRECTORY_SETGID_POLICY
    with pytest.raises(ValueError, match="caller opt-in"):
        unpack_bundle(tmp_path / "new.tar", new, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()
    unpack_bundle(tmp_path / "new.tar", new, tmp_path / "new", preserve_directory_setgid=True)
    assert (tmp_path / "new/execution").stat().st_mode & 0o7777 == 0o2770
    assert (tmp_path / "new/execution/database").read_bytes() == (source / "execution/database").read_bytes()
    assert (tmp_path / "new/execution/database").stat().st_mode & 0o7777 == 0o640
    assert (tmp_path / "new/execution/database").stat().st_uid == os.getuid()
    # An opted-in reader still accepts old manifests.
    unpack_bundle(tmp_path / "old.tar", old, tmp_path / "legacy", preserve_directory_setgid=True)


@pytest.mark.parametrize("name,mode", [("execution", 0o4770), ("execution", 0o1770), ("execution/database", 0o2640)])
def test_opt_in_rejects_other_special_bits(tmp_path, name, mode):
    source = tree(tmp_path)
    (source / name).chmod(mode)
    with pytest.raises(ValueError, match="special permission"):
        pack_bundle(source, tmp_path / "bad.tar", preserve_directory_setgid=True)
    assert not (tmp_path / "bad.tar").exists()


def test_archive_cannot_authorize_special_bits_or_restore_ownership(tmp_path):
    source = tree(tmp_path)
    archive = tmp_path / "bundle.tar"
    manifest = pack_bundle(source, archive, preserve_directory_setgid=True)
    # Ownership in a foreign archive is ignored, including names.
    with tarfile.open(archive) as original:
        contents = [(m, original.extractfile(m).read() if m.isfile() else None) for m in original]
    import io

    with tarfile.open(archive, "w") as out:
        for member, data in contents:
            member.uid = member.gid = 12345
            member.uname = member.gname = "foreign-owner"
            out.addfile(member, io.BytesIO(data) if data is not None else None)
    manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    unpack_bundle(archive, manifest, tmp_path / "owned", preserve_directory_setgid=True)
    assert (tmp_path / "owned/execution").stat().st_uid == os.getuid()
    assert (tmp_path / "owned/execution").stat().st_gid != 12345
    manifest["permission_policy"] = "arbitrary"
    with pytest.raises(ValueError, match="permission policy"):
        unpack_bundle(archive, manifest, tmp_path / "bad", preserve_directory_setgid=True)
    manifest.pop("permission_policy")
    with pytest.raises(ValueError, match="mode mismatch"):
        unpack_bundle(archive, manifest, tmp_path / "bad", preserve_directory_setgid=True)


def test_permission_downgrade_fails_before_publication(tmp_path, monkeypatch):
    from pathlib import Path

    source = tree(tmp_path)
    archive = tmp_path / "bundle.tar"
    manifest = pack_bundle(source, archive, preserve_directory_setgid=True)
    chmod = Path.chmod

    def stripped(path, mode, *args, **kwargs):
        return chmod(path, mode & 0o777, *args, **kwargs)

    monkeypatch.setattr(Path, "chmod", stripped)
    with pytest.raises(ValueError, match="did not preserve"):
        unpack_bundle(archive, manifest, tmp_path / "result", preserve_directory_setgid=True)
    assert not (tmp_path / "result").exists()
