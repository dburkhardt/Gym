"""Isolate a pinned Harbor single-step artifact-copy compatibility fix.

Never edit the installed dependency or recorded artifacts. Regrade must retain
symlink text until Docker restores it inside the container, as multi-step
Harbor regrade already does; host dereferencing is incorrect and unsafe.
"""

import hashlib
import shutil
from pathlib import Path


PINNED_REGRADE_SHA256 = "9d814a7949aea87dc2dc9a84e1d65fb66ac37484ebd59923320c14c281e271e5"


def stage_symlink_preserving_harbor(installed_package: Path, destination: Path) -> dict:
    source = Path(installed_package)
    destination = Path(destination)
    module = source / "trial/regrade.py"
    original = module.read_bytes()
    if hashlib.sha256(original).hexdigest() != PINNED_REGRADE_SHA256:
        raise ValueError("Unreviewed Harbor regrade source")
    old = b"            preserve_symlinks=False,\n"
    if original.count(old) != 1:
        raise ValueError("Expected exactly one single-step compatibility site")
    if destination.exists():
        raise ValueError("Compatibility overlay already exists")
    shutil.copytree(
        source, destination / "harbor", symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    changed = destination / "harbor/trial/regrade.py"
    changed.write_bytes(original.replace(old, b"            preserve_symlinks=True,\n"))
    hashes = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*")
        if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    for relative, digest in hashes.items():
        if (
            relative != "trial/regrade.py"
            and hashlib.sha256((destination / "harbor" / relative).read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Unintended Harbor package change")
    return {
        "original_package_files": hashes,
        "original_regrade_sha256": PINNED_REGRADE_SHA256,
        "patched_regrade_sha256": hashlib.sha256(changed.read_bytes()).hexdigest(),
        "change": "single-step seed preserves recorded symlinks like multi-step seed",
        "installed_dependency_modified": False,
        "recorded_artifacts_modified": False,
        "new_model_calls": 0,
    }
