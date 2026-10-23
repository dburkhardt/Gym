# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone sidecar helper for the opt-in immutable evaluation-input variant.

No Gym dependency: materialization copies this module into the pinned sidecar.
Source stability checks detect ordinary concurrent writes, not transactional
multi-file publication by an adversarial writer.
"""

import hashlib
import json
import os
import shutil
import stat
import time
from pathlib import Path


class SnapshotCaptureError(RuntimeError):
    """Operational input capture failed; no scientific score was produced."""


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise SnapshotCaptureError("Evaluation input capture deadline exhausted")


def _publish_manifest(destination: Path, manifest: dict[str, object]) -> None:
    temporary = destination / ".manifest.tmp"
    with temporary.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination / "manifest.json")
    fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)


def _inventory(root_fd: int, deadline: float) -> dict[str, tuple[int, ...]]:
    entries = {}
    for directory, dirs, files, fd in os.fwalk(".", dir_fd=root_fd, follow_symlinks=False):
        for name in dirs + files:
            _check_deadline(deadline)
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if (
                not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode))
                or info.st_nlink > 1
                and stat.S_ISREG(info.st_mode)
            ):
                raise SnapshotCaptureError("Snapshot source contains a link or unsupported entry")
            entries[str(Path(directory) / name)] = _identity(info)
    return entries


def _copy_file(source_fd: int, destination: Path, deadline: float) -> str:
    digest = hashlib.sha256()
    with destination.open("xb") as output:
        while chunk := os.read(source_fd, 1024 * 1024):
            _check_deadline(deadline)
            output.write(chunk)
            digest.update(chunk)
        output.flush()
        os.fsync(output.fileno())
    return digest.hexdigest()


def capture_inputs(source: Path, destination: Path, *, deadline: float) -> dict[str, object]:
    """Copy a public output tree, refusing detected mutation and all links.

    Retain a manifest on normal capture failures when storage permits; disk or
    publication errors fail closed. Callers must never evaluate a failed capture.
    """
    output = destination / "output"
    created = False
    files: dict[str, object] = {}
    manifest: dict[str, object] = {"status": "capturing", "files": files}
    root_fd = None
    try:
        if source != source.resolve() or source.is_symlink():
            raise SnapshotCaptureError("Snapshot source path contains a link")
        if destination.resolve().is_relative_to(source):
            raise SnapshotCaptureError("Snapshot destination must be outside source")
        destination.mkdir(parents=True, exist_ok=False, mode=0o700)
        created = True
        output.mkdir(mode=0o700)
        root_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        _check_deadline(deadline)
        before = _inventory(root_fd, deadline)
        for directory, dirs, names, fd in os.fwalk(".", dir_fd=root_fd, follow_symlinks=False):
            for name in dirs:
                (output / directory / name).mkdir(mode=0o700)
            for name in names:
                relative = str(Path(directory) / name)
                source_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
                try:
                    info = os.fstat(source_fd)
                    if not stat.S_ISREG(info.st_mode) or _identity(info) != before.get(relative):
                        raise SnapshotCaptureError("Snapshot source changed before copy")
                    target = output / relative
                    _check_deadline(deadline)
                    if shutil.disk_usage(destination).free < info.st_size + 1024 * 1024:
                        raise SnapshotCaptureError("Insufficient snapshot disk space")
                    digest = _copy_file(source_fd, target, deadline)
                    os.lseek(source_fd, 0, os.SEEK_SET)
                    check = hashlib.sha256()
                    while chunk := os.read(source_fd, 1024 * 1024):
                        _check_deadline(deadline)
                        check.update(chunk)
                    if _identity(os.fstat(source_fd)) != before[relative] or check.hexdigest() != digest:
                        raise SnapshotCaptureError("Snapshot source changed during copy")
                    copied_digest = hashlib.sha256()
                    with target.open("rb") as copied:
                        while chunk := copied.read(1024 * 1024):
                            _check_deadline(deadline)
                            copied_digest.update(chunk)
                    if copied_digest.hexdigest() != digest:
                        raise SnapshotCaptureError("Snapshot copy verification failed")
                    target.chmod(0o400)
                    files[relative] = {"sha256": digest, "bytes": info.st_size}
                finally:
                    os.close(source_fd)
        if (
            _inventory(root_fd, deadline) != before
            or _identity(os.stat(source, follow_symlinks=False))[:2] != _identity(os.fstat(root_fd))[:2]
        ):
            raise SnapshotCaptureError("Snapshot source tree changed during copy")
        for directory, _, _ in os.walk(output, topdown=False):
            _check_deadline(deadline)
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        _check_deadline(deadline)
        manifest["status"] = "complete"
        return manifest
    except (OSError, SnapshotCaptureError) as error:
        manifest.update(status="failed", failure_kind="evaluation_input_snapshot_failed", error=str(error))
        raise SnapshotCaptureError(str(error)) from error
    finally:
        if root_fd is not None:
            os.close(root_fd)
        try:
            if created:
                _publish_manifest(destination, manifest)
        except OSError as error:
            raise SnapshotCaptureError(
                f"Snapshot manifest publication failed: {error}; capture status: {manifest['status']}"
            ) from error
