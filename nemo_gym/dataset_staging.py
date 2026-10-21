# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checksum-verified publication of locally cached dataset files.

Transport and remote caching remain the downloader's responsibility. Interrupted
copies resume from a verified prefix; readers see only a fully verified final file.
Advisory locks coordinate cooperating writers on POSIX filesystems.
"""

import fcntl
import hashlib
import os
import re
import shutil
import stat
from pathlib import Path


_CHUNK_SIZE = 1024 * 1024


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _regular_fd(path: Path, flags: int) -> int:
    # A FIFO must not block before fstat can reject the non-regular file.
    fd = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError(f"Staging path must be a regular file: {path}")
    return fd


def _sync_parent(destination: Path) -> None:
    directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def stage_file(source: Path, destination: Path, *, expected_sha256: str | None = None) -> Path:
    """Atomically publish a cached file after checking its SHA-256 digest.

    With no supplied digest, hash the local source before copying. HF cache
    symlinks are allowed for the source. Destination and staging symlinks are
    refused. The destination's parent must be a trusted directory; this helper
    coordinates cooperating writers, not hostile directory mutation.

    A partial copy is retained on failure, keyed by the source digest. On retry,
    its prefix is compared with the source before appending. A corrupt prefix is
    discarded. Existing final files are replaced only after full verification. If a supplied
    digest already matches the destination, the source need not exist. Publication
    fsyncs the parent directory; filesystems must support directory fsync. A failure
    of that final fsync means the verified file is visible but crash durability is
    uncertain, so the error is propagated.
    """
    source, destination = Path(source), Path(destination)
    if expected_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 hex digest")
    digest = expected_sha256 or _hash(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(f".{destination.name}.stage.lock")
    partial = destination.with_name(f".{destination.name}.{digest}.partial")
    with os.fdopen(_regular_fd(lock, os.O_CREAT | os.O_RDWR), "rb+") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX)
        if destination.is_symlink():
            raise ValueError(f"Destination must not be a symlink: {destination}")
        existing_digest = None
        if destination.exists():
            with os.fdopen(_regular_fd(destination, os.O_RDONLY), "rb") as existing:
                existing_digest = hashlib.file_digest(existing, "sha256").hexdigest()
        if existing_digest == digest:
            # A prior publication may have failed after rename but before directory fsync.
            _sync_parent(destination)
            return destination
        with source.open("rb") as incoming, os.fdopen(_regular_fd(partial, os.O_CREAT | os.O_RDWR), "rb+") as staged:
            while block := staged.read(_CHUNK_SIZE):
                if incoming.read(len(block)) != block:
                    staged.seek(0)
                    staged.truncate()
                    incoming.seek(0)
                    break
            shutil.copyfileobj(incoming, staged, length=_CHUNK_SIZE)
            staged.flush()
            os.fsync(staged.fileno())
            staged.seek(0)
            actual = hashlib.file_digest(staged, "sha256").hexdigest()
            if actual != digest:
                raise ValueError(f"Dataset checksum mismatch for {source}: expected {digest}, got {actual}")
            # Refuse replacement if a staging pathname no longer names our open file.
            opened = os.fstat(staged.fileno())
            named = partial.lstat()
            if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
                raise ValueError(f"Staging file changed during copy: {partial}")
            if destination.is_symlink():
                raise ValueError(f"Destination became a symlink: {destination}")
            os.replace(partial, destination)
            _sync_parent(destination)
    return destination
