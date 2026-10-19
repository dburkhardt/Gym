# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a pinned, category-stratified GenoTEX pilot without downloading the full corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path


SOURCE_REV = "2ce91ce8fe326bb53a64a7cf25d57d961f75fe36"
DATA_REV = "9d50c9020256e8c943e02b6c0ad843017cd76cf8"
DATA_REPO = "Liu-Hy/GenoTEX"
OUTPUT = Path(__file__).parent / "data/genotex_benchmark.jsonl"


def fetch_json(url: str) -> object:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def dataset_tree() -> list[dict]:
    """Read all pages at an immutable revision."""
    url = f"https://huggingface.co/api/datasets/{DATA_REPO}/tree/{DATA_REV}?recursive=true&limit=1000"
    entries = []
    while url:
        with urllib.request.urlopen(url, timeout=120) as response:
            entries.extend(json.load(response))
            link = response.headers.get("Link")
            url = link.split(">")[0][1:] if link else None
    return entries


def download(entry: dict, root: Path) -> None:
    """Fetch immutable assets; verify LFS SHA256 or Git blob SHA1 before atomic rename."""
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe dataset path")
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)

    def valid(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size != entry["size"]:
            return False
        data = path.read_bytes()
        if entry.get("lfs"):
            return hashlib.sha256(data).hexdigest() == entry["lfs"]["oid"]
        return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == entry["oid"]

    if valid(target):
        return
    temporary = target.with_name(target.name + ".partial")
    url = f"https://huggingface.co/datasets/{DATA_REPO}/resolve/{DATA_REV}/{urllib.parse.quote(entry['path'])}"
    with urllib.request.urlopen(url, timeout=300) as source, temporary.open("wb") as output:
        while chunk := source.read(1024 * 1024):
            output.write(chunk)
    if not valid(temporary):
        raise ValueError(f"Checksum mismatch: {relative}")
    temporary.replace(target)


def select_pairs(metadata: dict, entries: list[dict]) -> tuple[list[dict], list[dict]]:
    """Choose 2 unconditional, 2 age/gender, 1 two-step pair by total raw input bytes."""
    files = {e["path"]: e for e in entries if e["type"] == "file"}
    sizes = {}
    for path, entry in files.items():
        parts = path.split("/")
        if parts[:2] == ["input", "GEO"] and len(parts) > 3:
            sizes[parts[2]] = sizes.get(parts[2], 0) + entry["size"]
    xena = {p.split("/")[2] for p in files if p.startswith("output/preprocess/") and p.endswith("/Xena.csv")}
    eligible, excluded = [], []
    for trait in sorted(metadata):
        for condition in [None, *metadata[trait]["conditions"]]:
            traits = sorted({trait} | ({condition} if condition not in (None, "Age", "Gender") else set()))
            reference = f"output/regress/{trait}/significant_genes_condition_{condition}.json"
            category = (
                "unconditional"
                if condition is None
                else "conditional"
                if condition in ("Age", "Gender")
                else "two-step"
            )
            item = {
                "trait": trait,
                "condition": condition,
                "traits": traits,
                "category": category,
                "task_id": f"{trait}__{condition}",
            }
            reason = None
            if reference not in files:
                reason = "No published regression reference; unsuitable for three-stage pilot"
            elif any(t in xena or t not in sizes for t in traits):
                reason = "TCGA mapping not supported by GEO-only pilot preparation"
            if reason:
                excluded.append({**item, "reason": reason})
            else:
                eligible.append({**item, "input_bytes": sum(sizes[t] for t in traits)})
    chosen = []
    for category, count in [("unconditional", 2), ("conditional", 2), ("two-step", 1)]:
        group = sorted(
            [p for p in eligible if p["category"] == category], key=lambda p: (p["input_bytes"], p["task_id"])
        )
        chosen.extend(group[:count])
        excluded.extend({**p, "reason": "Beyond deterministic category pilot quota"} for p in group[count:])
    return chosen, excluded


def prepare(root: Path | None = None) -> Path:
    """Download selected raw inputs and private references, then write Gym records."""
    root = (root or Path(os.environ.get("GENOTEX_DATA_ROOT", "~/.cache/genotex"))).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    upstream = root / "upstream"
    if not upstream.exists():
        subprocess.run(["git", "clone", "https://github.com/Liu-Hy/GenoTEX.git", str(upstream)], check=True)
    subprocess.run(["git", "checkout", "--detach", SOURCE_REV], cwd=upstream, check=True)
    entries = dataset_tree()
    for entry in entries:
        if entry["path"].startswith("metadata/") and entry["type"] == "file":
            download(entry, root / "assets")
    metadata = json.loads((root / "assets/metadata/task_info.json").read_text())
    selected, excluded = select_pairs(metadata, entries)
    traits = {t for p in selected for t in p["traits"]}
    regression = {f"output/regress/{p['trait']}/significant_genes_condition_{p['condition']}.json" for p in selected}
    for entry in entries:
        p = entry["path"]
        if entry["type"] == "file" and (
            p in regression
            or any(p.startswith(f"input/GEO/{t}/") or p.startswith(f"output/preprocess/{t}/") for t in traits)
        ):
            download(entry, root / "assets")
    selection = {
        "source_revision": SOURCE_REV,
        "dataset_revision": DATA_REV,
        "selection_rule": "GEO-only; published regression reference; 2 unconditional, 2 age/gender, 1 two-step; ascending raw bytes then task ID",
        "selected": selected,
        "excluded": excluded,
    }
    (root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": f"Complete the GenoTEX analysis for trait {p['trait']}, condition {p['condition']}.",
                    }
                ]
            },
            "verifier_metadata": p,
            "task_id": p["task_id"],
        }
        for p in selected
    ]
    OUTPUT.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "tasks.jsonl").write_text(OUTPUT.read_text())
    (root / "tasks-smoke.jsonl").write_text(json.dumps(rows[0]) + "\n")
    (root / "tasks-remaining.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[1:]))
    return OUTPUT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    print(prepare(args.data_root))
