"""Validate the creator's released tables; never imply complete evaluation inputs."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def validate_release(root, *, expected_manifest_sha256):
    root = Path(root).resolve()
    manifest_path = root / "MANIFEST.json"
    if manifest_path.is_symlink() or sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("Creator manifest binding failed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("release") != "AgentIdeaBench" or not manifest.get("files"):
        raise ValueError("Not an AgentIdeaBench release manifest")
    seen = set()
    verified = []
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in seen:
            raise ValueError("Unsafe or duplicate release path")
        seen.add(relative.as_posix())
        path = root / relative
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError("Release symlinks are not accepted")
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError("Missing or escaped release file")
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Release checksum changed: {relative}")
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            columns = next(reader)
            if columns != entry["columns"]:
                raise ValueError(f"Release schema changed: {relative}")
            count = 0
            for row in reader:
                if len(row) != len(columns):
                    raise ValueError(f"Malformed CSV row: {relative}")
                count += 1
        if count != entry["rows"]:
            raise ValueError(f"Release row count changed: {relative}")
        verified.append({"path": relative.as_posix(), "rows": count, "sha256": entry["sha256"]})
    return {
        "status": "released_tables_verified_not_complete_benchmark_inputs",
        "manifest_sha256": expected_manifest_sha256,
        "files": verified,
        "dropped_columns": manifest.get("dropped_columns", []),
        "redaction": manifest.get("redaction"),
        "withheld_text": manifest.get("withheld_text"),
        "limitations": [
            "Rows include results and derived tables; their sum is not a task count",
            "Checksums establish the supplied manifest binding, not corpus text availability",
            "Static references, active retrieval and critic slots require separate qualification",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(validate_release(args.root, expected_manifest_sha256=args.manifest_sha256), indent=2))
