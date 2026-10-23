# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize only the pinned Git tree; ignored checkout files never enter imports."""

import io
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


CREATOR_REVISION = "35eed91676062d98d98562556ec6cce326fd7d93"


def validate_creator_root(source: Path) -> dict[str, str]:
    """Preflight the same pinned Git prerequisites required by snapshot creation."""
    try:
        revision = (
            subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], timeout=60, stderr=subprocess.PIPE
            )
            .decode()
            .strip()
        )
        changes = subprocess.check_output(
            ["git", "-C", str(source), "diff", "HEAD", "--name-only"], timeout=60, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError("creator_root must be the original pinned Git checkout, not an exported snapshot") from exc
    if revision != CREATOR_REVISION:
        raise ValueError("Creator revision differs from pinned source")
    if changes.strip():
        raise ValueError("Creator tracked source has modifications")
    return {"creator_root": str(source.resolve()), "revision": revision, "tracked_source": "unchanged"}


def snapshot_creator(source: Path, destination: Path) -> Path:
    """Read the immutable creator tree, refusing symlinks and nonregular entries."""
    validate_creator_root(source)

    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(source), *args], timeout=60)

    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", CREATOR_REVISION))) as archive:
        for entry in archive:
            relative = PurePosixPath(entry.name)
            if relative.is_absolute() or ".." in relative.parts or not (entry.isfile() or entry.isdir()):
                raise ValueError("Unsafe creator archive entry")
            target = destination.joinpath(*relative.parts)
            if entry.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(entry) as content:
                    target.write_bytes(content.read())
                target.chmod(entry.mode & 0o777)
    return destination


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Model-free creator Git preflight before native service startup")
    parser.add_argument("--creator-root", type=Path, required=True)
    print(json.dumps(validate_creator_root(parser.parse_args().creator_root), sort_keys=True))
