# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Corral's required scientific state over shared verified bundle transport."""

from pathlib import Path

from nemo_gym.state_bundle import pack_bundle, unpack_bundle


def pack_state(source: Path, destination: Path) -> dict:
    if not (source / "solver-result.json").is_file() or not list(source.glob("state/**/commits.sqlite3")):
        raise ValueError("Corral state is missing creator persistence or submission")
    return pack_bundle(source, destination)


def unpack_state(archive_path: Path, manifest: dict, destination: Path) -> None:
    files = manifest["entries"]
    if "solver-result.json" not in files or not any(name.endswith("/commits.sqlite3") for name in files):
        raise ValueError("Corral state is missing creator persistence or submission")
    unpack_bundle(archive_path, manifest, destination)
