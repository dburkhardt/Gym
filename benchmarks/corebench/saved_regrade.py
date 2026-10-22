"""Prepare CORE's unchanged verifier for Harbor recorded-artifact replay."""
from pathlib import Path
import hashlib
import shutil
import tomllib


def prepare_task(original: Path, destination: Path) -> dict:
    """Copy the task, selecting separate verification; preserve every scorer byte.

    Harbor restores declared artifacts to their original absolute paths, so the
    original /workspace/report.json reader must remain unchanged. This helper
    never supplies a solution or modifies the original live task.
    """
    original, destination = Path(original), Path(destination)
    config = tomllib.loads((original / "task.toml").read_text())
    if "/workspace" not in config.get("artifacts", []):
        raise ValueError("CORE replay requires declared /workspace artifacts")
    if (original / "solution").exists():
        raise ValueError("Operational solution tasks are not model replay inputs")
    text = (original / "task.toml").read_text()
    if config.get("verifier", {}).get("environment_mode") not in (None, "shared"):
        raise ValueError("Expected original shared verifier")
    if "environment_mode" in text:
        raise ValueError("Explicit verifier mode requires separate review")
    if text.count("[verifier]\n") != 1:
        raise ValueError("Expected one verifier section")
    shutil.copytree(original, destination)
    (destination / "task.toml").write_text(
        text.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n')
    )
    hashes = {}
    for path in (original / "tests").rglob("*"):
        if path.is_file():
            relative = path.relative_to(original)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if hashlib.sha256((destination / relative).read_bytes()).hexdigest() != digest:
                raise ValueError("Verifier changed during preparation")
            hashes[str(relative)] = digest
    return {"verifier_files": hashes, "new_solver_calls": 0,
            "artifact_restoration": "Harbor restores declared /workspace to original path"}
