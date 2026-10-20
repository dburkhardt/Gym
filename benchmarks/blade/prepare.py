# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin BLADE's real native source and separate solver inputs from scorer references."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_gym.dataset_staging import stage_file


PUBLIC_DATA = {"data.csv", "info.json"}


def stage(source: Path, destination: Path) -> Path:
    """Stage the existing three-task selection; never invent a scalar scientific reward."""
    source, destination = source.resolve(), destination.resolve()
    manifest = json.loads(Path(__file__).with_name("frozen_sources.json").read_text())
    files = manifest["files"]
    # Validate every pinned file before publishing any selected data.
    for relative, checksum in files.items():
        path = source / relative
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"BLADE pinned source mismatch: {relative}")
    expected_solver = {
        relative
        for relative in files
        if (
            relative.startswith("blade_bench/")
            and relative.endswith(".py")
            and not relative.startswith("blade_bench/datasets/")
        )
        or relative in ("pyproject.toml", "LICENSE")
    }
    expected_solver.update(
        f"blade_bench/datasets/{dataset}/{name}" for dataset in manifest["selected_datasets"] for name in PUBLIC_DATA
    )
    solver_root = destination / "solver"
    for path in solver_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("BLADE solver staging must not contain symlinks")
        if path.is_file():
            relative = str(path.relative_to(solver_root))
            if path.name.endswith(".stage.lock") and path.name.startswith("."):
                relative = str(path.relative_to(solver_root).with_name(path.name[1:-11]))
            if relative not in expected_solver:
                raise ValueError(f"Unexpected file in solver staging: {path}")
    rows = []
    for dataset in manifest["selected_datasets"]:
        dataset_root = f"blade_bench/datasets/{dataset}/"
        info = json.loads((source / dataset_root / "info.json").read_text())
        rows.append(
            {
                "task_id": f"blade::{dataset}",
                "uuid": f"blade::{dataset}",
                "responses_create_params": {"input": []},
                "verifier_metadata": {
                    "dataset": dataset,
                    "source_commit": manifest["source_commit"],
                    "native_info": info,
                },
            }
        )
    for relative, checksum in files.items():
        parts = Path(relative).parts
        if relative.startswith("blade_bench/datasets/"):
            # Every dataset is explicitly selected; no hidden reference files in solver context.
            if len(parts) != 4 or parts[2] not in manifest["selected_datasets"]:
                continue
            role = "solver" if parts[3] in PUBLIC_DATA else "references"
            target = destination / role / relative
        elif relative.startswith("blade_bench/") and relative.endswith(".py"):
            target = destination / "solver" / relative
        elif relative in ("pyproject.toml", "LICENSE"):
            target = destination / "solver" / relative
        else:
            continue
        stage_file(source / relative, target, expected_sha256=checksum)
    with TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        data = temporary / "benchmark.jsonl"
        data.write_text("".join(json.dumps(row) + "\n" for row in rows))
        receipt = temporary / "source.json"
        receipt.write_text(
            json.dumps(
                {
                    "source_commit": manifest["source_commit"],
                    "selected_datasets": manifest["selected_datasets"],
                    "source_files": files,
                    "execution_status": "native_transport_and_harbor_hooks_pending",
                },
                indent=2,
            )
        )
        stage_file(receipt, destination / "source.json")
        output = destination / "benchmark.jsonl"
        stage_file(data, output)
    return output


def prepare() -> Path:
    """Native preparation entrypoint; source is the pinned public BLADE checkout."""
    return stage(Path(os.environ["BLADE_SOURCE"]), Path(__file__).with_name("data"))


if __name__ == "__main__":
    prepare()
