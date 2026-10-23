# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen verified metadata and prerequisite inventory, not execution qualification.

No package installation, archive download, provider dispatch or generated program
execution occurs here. Solver-visible task fields follow creator format_task_dict.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_gym.dataset_staging import stage_file


MANIFEST = Path(__file__).with_name("frozen50.json")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def contained(root, relative, directory=False):
    root = Path(root).absolute()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Invalid asset path")
    path = root / relative
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Asset symlinks are unsupported")
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"Missing required creator asset: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Asset escapes root")
    return path


def load_frozen(source, parquet):
    import pandas as pd

    manifest = json.loads(MANIFEST.read_text())
    for relative, expected in manifest["source_files"].items():
        if digest(contained(source, relative)) != expected:
            raise ValueError(f"Creator source changed: {relative}")
    if digest(Path(parquet)) != manifest["parquet_sha256"]:
        raise ValueError("Frozen verified parquet changed")
    frame = pd.read_parquet(parquet)
    rows = []
    for selected in manifest["tasks"]:
        row = frame.iloc[selected["row_index"]].to_dict()
        actual = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if actual != selected["metadata_sha256"]:
            # The byte-pinned Parquet's nullable string cells were originally
            # serialized as NaN. Some pandas/Arrow versions expose them as None.
            # Compare the legacy representation only; never alter source values
            # returned to formatting or accept a changed non-null value.
            legacy = {key: float("nan") if value is None else value for key, value in row.items()}
            actual = hashlib.sha256(json.dumps(legacy, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if actual != selected["metadata_sha256"] or row["instance_id"] != selected["instance_id"]:
            raise ValueError("Frozen task identity changed")
        rows.append((selected, row))
    return manifest, rows


def format_task(example, use_knowledge=False):
    # Preserve run_infer.format_task_dict with the creator's sandbox-relative root.
    task = {
        "task_inst": example["task_inst"],
        "dataset_path": "benchmark/datasets/" + example["dataset_folder_tree"].split("\n")[0][4:],
        "dataset_folder_tree": example["dataset_folder_tree"],
        "dataset_preview": example["dataset_preview"],
        "output_fname": example["output_fname"],
    }
    if use_knowledge:
        task["domain_knowledge"] = example["domain_knowledge"]
    return task


def enumerated_paths(tree):
    """Parse concrete paths in the frozen creator tree, without inventing omissions."""
    parents = {}
    result = []
    for line in tree.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"\|(-{2}(?:--)*?) (.+)", line)
        if not match:
            raise ValueError("Unsupported frozen folder-tree syntax")
        depth = len(match[1]) // 2 - 1
        name = match[2]
        if name in {"...", "…"}:
            continue
        if depth and depth - 1 not in parents:
            raise ValueError("Invalid folder-tree nesting")
        path = (parents[depth - 1] if depth else Path()) / name.rstrip("/")
        if Path(name).is_absolute() or "/" in name.rstrip("/") or ".." in path.parts:
            raise ValueError("Invalid folder-tree path")
        is_directory = name.endswith("/")
        result.append(("datasets/" + path.as_posix(), is_directory))
        parents = {key: value for key, value in parents.items() if key < depth}
        if is_directory:
            parents[depth] = path
    return result


def prepare(
    source, parquet, assets, destination, use_knowledge=False, asset_manifest=None, asset_manifest_sha256=None
):
    manifest, frozen = load_frozen(source, parquet)
    assets = Path(assets)
    expected_assets = None
    if (asset_manifest is None) != (asset_manifest_sha256 is None):
        raise ValueError("Asset manifest and its trusted checksum must be supplied together")
    if asset_manifest is not None:
        if digest(Path(asset_manifest)) != asset_manifest_sha256:
            raise ValueError("Asset manifest checksum changed")
        expected_assets = json.loads(Path(asset_manifest).read_text())
        if not isinstance(expected_assets, dict) or not expected_assets:
            raise ValueError("Expected asset manifest must be a nonempty path-to-SHA256 map")
        for relative, expected in expected_assets.items():
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("Invalid expected asset checksum")
            if digest(contained(assets, relative)) != expected:
                raise ValueError(f"Asset checksum changed: {relative}")
    inventory = {}
    rows = []
    for selected, example in frozen:
        task = format_task(example, use_knowledge)
        dataset_relative = task["dataset_path"].removeprefix("benchmark/")
        for relative, is_directory in enumerated_paths(example["dataset_folder_tree"]):
            contained(assets, relative, directory=is_directory)
        directory = contained(assets, dataset_relative, directory=True)
        files = sorted(directory.rglob("*"))
        if not any(p.is_file() for p in files):
            raise ValueError(f"Empty creator dataset: {dataset_relative}")
        for path in files:
            if path.is_symlink():
                raise ValueError("Asset symlinks are unsupported")
            if path.is_file():
                relative = path.relative_to(assets).as_posix()
                inventory[relative] = digest(contained(assets, relative))
        for relative in (
            "gold_programs/" + example["gold_program_name"],
            "eval_programs/" + example["eval_script_name"],
        ):
            inventory[relative] = digest(contained(assets, relative))
        rows.append(
            {
                "task_id": str(selected["instance_id"]),
                "source_row_index": selected["row_index"],
                "creator_task": task,
                "verifier_metadata": {
                    "instance_id": selected["instance_id"],
                    "gold_program_name": example["gold_program_name"],
                    "eval_script_name": example["eval_script_name"],
                    "metadata_sha256": selected["metadata_sha256"],
                },
            }
        )
    if expected_assets is not None and any(expected_assets.get(path) != value for path, value in inventory.items()):
        raise ValueError("Expected asset manifest does not cover every selected asset")
    receipt = {
        "version": 1,
        "source_revision": manifest["source_revision"],
        "dataset_revision": manifest["dataset_revision"],
        "manifest_sha256": digest(MANIFEST),
        "assets": inventory,
        "supplied_asset_manifest_sha256": asset_manifest_sha256,
        "supplied_asset_checksums_verified": expected_assets is not None,
        "status": "required_paths_present_not_runtime_or_archive_provenance_qualified",
        "limitations": [
            "Verified archive publisher checksum is not available in retained evidence",
            "Evaluation-script auxiliary references and runtime dependencies still require isolated qualification",
            "Never mount this inventory, gold programs or evaluation files into solver workspace",
        ],
    }
    destination = Path(destination)
    # Fresh publication directory preserves prior receipts and task bytes.
    # A staging error may leave partial owned output for inspection.
    destination.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "tasks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (tmp / "asset-preflight.json").write_text(json.dumps(receipt, indent=2) + "\n")
        stage_file(tmp / "asset-preflight.json", destination / "asset-preflight.json")
        return stage_file(tmp / "tasks.jsonl", destination / "tasks.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "parquet", "assets", "destination"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--use-knowledge", action="store_true")
    parser.add_argument("--asset-manifest", type=Path)
    parser.add_argument("--asset-manifest-sha256")
    print(prepare(**vars(parser.parse_args())))
