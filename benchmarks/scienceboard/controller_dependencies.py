"""Verify supplied controller artifacts and publish an offline pip lock; no downloads."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from packaging.utils import parse_wheel_filename

from benchmarks.scienceboard.prepare import MANIFEST, checked_source
from nemo_gym.dataset_staging import stage_file

LOCK = Path(__file__).with_name("controller-dependencies.json")


def verify_artifacts(root: Path, expected: dict[str, str]) -> dict[str, Path]:
    return {name: checked_source(root, name, digest) for name, digest in expected.items()}


def offline_requirements(wheels: dict[str, str]) -> str:
    rows = []
    names = set()
    for filename, digest in sorted(wheels.items()):
        name, version, _, _ = parse_wheel_filename(filename)
        if name in names:
            raise ValueError("Duplicate distribution in controller lock")
        names.add(name)
        rows.append(f"{name}=={version} --hash=sha256:{digest}")
    return "\n".join(rows) + "\n"


def prepare(source: Path, wheels: Path, archives: Path, output: Path) -> Path:
    lock = json.loads(LOCK.read_text())
    creator = json.loads(MANIFEST.read_text())
    if lock["source_revision"] != creator["source_revision"]:
        raise ValueError("Controller lock source revision mismatch")
    for name, digest in creator["files"].items():
        checked_source(source, name, digest)
    verify_artifacts(wheels, lock["wheels"])
    os_files = {p["artifact_filename"]: p["SHA256"] for p in lock["os_packages"]}
    verify_artifacts(archives, os_files)
    if output.exists():
        raise FileExistsError("Use a fresh controller preparation directory")
    requirements = offline_requirements(lock["wheels"])
    with TemporaryDirectory() as tmp:
        staged = Path(tmp) / "requirements.txt"
        staged.write_text(requirements)
        stage_file(staged, output / "requirements.txt")
        receipt = Path(tmp) / "receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "scope": lock["scope"],
                    "gui_qualified": False,
                    "lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
                    "source_revision": lock["source_revision"],
                    "wheels": str(wheels.absolute()),
                    "archives": str(archives.absolute()),
                    "wheel_count": len(lock["wheels"]),
                    "os_archive_count": len(os_files),
                },
                indent=2,
            )
            + "\n"
        )
        stage_file(receipt, output / "receipt.json")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "wheels", "archives", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.wheels, args.archives, args.output)


if __name__ == "__main__":
    main()
