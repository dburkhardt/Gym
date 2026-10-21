# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hash-verified regular-file state transport for separate execution phases."""

import hashlib
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory


MAX_BYTES = 1024**3
MAX_MEMBERS = 100_000


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


DIRECTORY_SETGID_POLICY = "preserve-directory-setgid-v1"


def _mode_allowed(mode: int, kind: str, preserve_directory_setgid: bool) -> bool:
    allowed = 0o777 | (0o2000 if preserve_directory_setgid and kind == "directory" else 0)
    return type(mode) is int and mode >= 0 and mode & ~allowed == 0


def pack_bundle(source: Path, destination: Path, *, preserve_directory_setgid: bool = False) -> dict:
    """Preserve bytes and ordinary modes; optionally preserve directory setgid only.

    The opt-in policy rejects every other special bit. Default legacy bundles
    continue stripping special bits. Ownership is never archived or restored.
    """
    if source.is_symlink() or not source.is_dir():
        raise ValueError("State must be a real directory")
    entries = {}
    total = 0
    for path in sorted(source.rglob("*")):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("State contains a link or special file")
        entry = {"kind": "directory" if path.is_dir() else "file", "mode": path.stat().st_mode & 0o777}
        if preserve_directory_setgid:
            entry["mode"] = path.stat().st_mode & 0o7777
            if not _mode_allowed(entry["mode"], entry["kind"], True):
                raise ValueError("Unsupported state special permission bits")
        if path.is_file():
            entry.update(size=path.stat().st_size, sha256=_sha(path))
            total += entry["size"]
        entries[path.relative_to(source).as_posix()] = entry
    if total > MAX_BYTES or len(entries) > MAX_MEMBERS:
        raise ValueError("State exceeds bundle limits")
    if destination.exists():
        raise FileExistsError(destination)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / "bundle.tar"
        with tarfile.open(staged, "w") as archive:
            for name, entry in entries.items():
                info = tarfile.TarInfo(name)
                info.mode = entry["mode"]
                if entry["kind"] == "directory":
                    info.type = tarfile.DIRTYPE
                    archive.addfile(info)
                else:
                    info.size = entry["size"]
                    with (source / name).open("rb") as stream:
                        archive.addfile(info, stream)
        manifest = {"schema_version": 1, "entries": entries, "archive_sha256": _sha(staged)}
        if preserve_directory_setgid:
            manifest["permission_policy"] = DIRECTORY_SETGID_POLICY
        # Catch source mutation during publication before making the bundle visible.
        _validate(staged, manifest)
        os.replace(staged, destination)
    return manifest


def _validate(path: Path, manifest: dict) -> None:
    if manifest.get("schema_version") != 1 or path.stat().st_size > MAX_BYTES + MAX_MEMBERS * 1024:
        raise ValueError("Invalid state bundle version or size")
    if _sha(path) != manifest["archive_sha256"]:
        raise ValueError("State archive checksum mismatch")
    policy = manifest.get("permission_policy")
    if policy not in (None, DIRECTORY_SETGID_POLICY):
        raise ValueError("Unsupported state permission policy")
    entries = manifest["entries"]
    if len(entries) > MAX_MEMBERS:
        raise ValueError("Too many state members")
    total = 0
    with tarfile.open(path) as archive:
        members = []
        for member in archive:
            if len(members) >= MAX_MEMBERS:
                raise ValueError("Too many archive members")
            members.append(member)
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(entries):
            raise ValueError("State archive file set mismatch")
        for member in members:
            name = PurePosixPath(member.name)
            entry = entries[member.name]
            if name.is_absolute() or ".." in name.parts or str(name) != member.name or member.name in ("", "."):
                raise ValueError("Unsafe state archive member")
            if any(
                entries.get(str(parent), {}).get("kind") != "directory"
                for parent in name.parents
                if str(parent) != "."
            ):
                raise ValueError("State parent collision or missing directory")
            if member.mode != entry["mode"] or not _mode_allowed(
                member.mode, entry["kind"], policy == DIRECTORY_SETGID_POLICY
            ):
                raise ValueError("State member mode mismatch")
            if member.isdir() and entry["kind"] == "directory":
                continue
            if not member.isfile() or entry["kind"] != "file" or member.size != entry["size"]:
                raise ValueError("Unsafe state member type or size")
            total += member.size
            if total > MAX_BYTES:
                raise ValueError("State exceeds bundle limits")
            stream = archive.extractfile(member)
            if stream is None or hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                raise ValueError("State file checksum mismatch")


def unpack_bundle(
    archive_path: Path, manifest: dict, destination: Path, *, preserve_directory_setgid: bool = False
) -> None:
    """Reject partial or unsafe state before atomically publishing a new tree."""
    if manifest.get("permission_policy") == DIRECTORY_SETGID_POLICY and not preserve_directory_setgid:
        raise ValueError("Directory setgid preservation requires caller opt-in")
    _validate(archive_path, manifest)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / "state"
        staged.mkdir()
        with tarfile.open(archive_path) as archive:
            for index, member in enumerate(archive):
                if index >= MAX_MEMBERS:
                    raise ValueError("Too many archive members")
                # Recheck before writes: the archive may have changed after validation.
                name = PurePosixPath(member.name)
                if (
                    name.is_absolute()
                    or ".." in name.parts
                    or str(name) != member.name
                    or member.name in ("", ".")
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError("Unsafe changed state archive member")
                entry = manifest["entries"].get(member.name)
                if (
                    entry is None
                    or member.mode != entry["mode"]
                    or (member.isdir() != (entry["kind"] == "directory"))
                    or (member.isfile() and member.size != entry.get("size"))
                ):
                    raise ValueError("Changed state archive member")
                target = staged / member.name
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, target.open("xb") as out:
                        shutil.copyfileobj(source, out)
            actual = {path.relative_to(staged).as_posix(): path for path in staged.rglob("*")}
            if set(actual) != set(manifest["entries"]):
                raise ValueError("Extracted state file set mismatch")
            for name, entry in manifest["entries"].items():
                path = actual[name]
                if entry["kind"] == "directory":
                    if not path.is_dir():
                        raise ValueError("Extracted state directory mismatch")
                elif not path.is_file() or path.stat().st_size != entry["size"] or _sha(path) != entry["sha256"]:
                    raise ValueError("Extracted state checksum mismatch")
            for name, entry in sorted(manifest["entries"].items(), key=lambda item: len(item[0]), reverse=True):
                target = staged / name
                target.chmod(entry["mode"])
                if target.stat().st_mode & 0o7777 != entry["mode"]:
                    raise ValueError("Filesystem did not preserve state permissions")
        os.replace(staged, destination)
