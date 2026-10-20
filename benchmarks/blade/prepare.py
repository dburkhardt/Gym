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


TOKENIZER_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
TOKENIZER_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
TOKENIZER_FILENAME = hashlib.sha1(TOKENIZER_URL.encode()).hexdigest()


def stage_tokenizer(source: Path, destination: Path) -> Path:
    """Pin the creator's tokenizer asset for network-disabled native execution."""
    return stage_file(source, destination / "runtime/tiktoken" / TOKENIZER_FILENAME, expected_sha256=TOKENIZER_SHA256)


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
    expected_solver.add("blade_bench/datasets/fertility/info.json")
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
                "instance_id": f"blade::{dataset}",
                "agent_ref": {"name": "harbor_agent"},
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
        if relative == "blade_bench/datasets/fertility/info.json":
            target = destination / "solver" / relative
        elif relative.startswith("blade_bench/datasets/"):
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
    for dataset in manifest["selected_datasets"]:
        task = destination / "tasks" / dataset
        with TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            for name, text in {
                "instruction.md": dataset + "\n",
                "tests/test.sh": "#!/bin/sh\necho Native_BLADE_custom_verifier_required >&2\nexit 1\n",
                "environment/Dockerfile": "FROM sha256:2d642ec0d5e668a7194e78665c58e2b42a7ef9b5b3dbb8ee3ecd0a3fbc4da829\nRUN mkdir -p /workspace\n",
                "task.toml": 'version = "1.0"\nartifacts = ["/workspace"]\n[agent]\ntimeout_sec = 3720\n[verifier]\ntimeout_sec = 3720\n[environment]\ndocker_image = "sha256:2d642ec0d5e668a7194e78665c58e2b42a7ef9b5b3dbb8ee3ecd0a3fbc4da829"\ncpus = 1\nmemory_mb = 128\nstorage_mb = 1024\nallow_internet = false\n',
            }.items():
                file = tmp / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(text)
                stage_file(file, task / name)
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
                    "execution_status": "native_hooks_implemented_runtime_acceptance_pending",
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
    destination = Path(__file__).with_name("data")
    result = stage(Path(os.environ["BLADE_SOURCE"]), destination)
    if os.environ.get("BLADE_TOKENIZER_ASSET"):
        stage_tokenizer(Path(os.environ["BLADE_TOKENIZER_ASSET"]), destination)
    else:
        import urllib.request

        with TemporaryDirectory() as directory:
            asset = Path(directory) / "cl100k_base.tiktoken"
            with urllib.request.urlopen(TOKENIZER_URL, timeout=30) as response:
                asset.write_bytes(response.read(4 * 1024 * 1024))
            stage_tokenizer(asset, destination)
    return result


if __name__ == "__main__":
    prepare()
