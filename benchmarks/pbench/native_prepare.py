# SPDX-License-Identifier: Apache-2.0
"""Derive runtime-only Harbor overlays from checksum-pinned scientific inputs."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.pbench.prepare import prepare as prepare_preserved
from nemo_gym.dataset_staging import stage_file


IMAGE = "sha256:f85315b46b3aae73f602ac2cf5270b0ea8fb8c05636f056dc1c5643ef1d48ece"


def native_prepare(source: Path, destination: Path, size: int = 5) -> Path:
    with TemporaryDirectory() as temporary:
        staged = Path(temporary) / "preserved"
        dataset = prepare_preserved(source, staged, size)
        receipt = json.loads((staged / "provenance.json").read_text())
        receipt["runtime_image"] = IMAGE
        receipt["runtime_overlay_hashes"] = {}
        destination = destination.absolute()
        if any(path.is_symlink() for path in (destination, *destination.parents)):
            raise ValueError("P-Bench destination must not alias another directory")
        for existing in (destination / "tasks").rglob("*"):
            if existing.is_symlink():
                raise ValueError("P-Bench destination contains a symlink")
            if existing.is_file() and not existing.name.endswith(".stage.lock"):
                relative = existing.relative_to(destination)
                if not (staged / relative).is_file():
                    raise ValueError("P-Bench destination contains unrecorded task files")
        for task in sorted((staged / "tasks").iterdir()):
            # Keep all scientific inputs unchanged; only pin the already built
            # runtime and declare complete workspace capture for native regrade.
            dockerfile = task / "environment/Dockerfile"
            lines = dockerfile.read_text().splitlines()
            if lines[0] != "FROM broad-pbench-runtime:20261020-f85315b":
                raise ValueError("Unexpected preserved P-Bench runtime base")
            # Harbor's supported prebuilt-image path uploads environment/ to
            # workdir when no Dockerfile exists. Local image IDs are not valid
            # BuildKit FROM repository references.
            old_destination = destination / dockerfile.relative_to(staged)
            if old_destination.exists():
                allowed = {dockerfile.read_text(), "\n".join([f"FROM {IMAGE}", *lines[1:]]) + "\n"}
                if old_destination.read_text() not in allowed:
                    raise ValueError("Refusing to replace unrecognized P-Bench build overlay")
                old_destination.unlink()
            dockerfile.unlink()
            config = task / "task.toml"
            text = config.read_text().replace(
                "[environment]\n", f'[environment]\ndocker_image = "{IMAGE}"\nworkdir = "/workspace"\n'
            )
            config.write_text('artifacts = ["/workspace"]\n' + text)
            receipt["runtime_overlay_hashes"][task.name] = {
                "task.toml": hashlib.sha256(config.read_bytes()).hexdigest()
            }
            for file in task.rglob("*"):
                if file.is_file() and not file.name.endswith(".stage.lock"):
                    published = stage_file(file, destination / file.relative_to(staged))
                    if file.suffix == ".sh":
                        published.chmod(0o755)
        receipt["status"] = "native runtime overlay prepared; actual runtime acceptance recorded separately"
        proof = Path(temporary) / "native-provenance.json"
        proof.write_text(json.dumps(receipt, indent=2) + "\n")
        stage_file(proof, destination / "provenance.json")
        return stage_file(dataset, destination / "benchmark.jsonl")


def prepare() -> Path:
    return native_prepare(
        Path(os.environ["PBENCH_PREPARED_SOURCE"]),
        Path(__file__).parent / "data",
        int(os.environ.get("PBENCH_SAMPLE_SIZE", "5")),
    )


if __name__ == "__main__":
    print(prepare())
