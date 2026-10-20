# SPDX-License-Identifier: Apache-2.0
"""Stage the frozen CORE-Bench medium capsule and recovered immutable Python runtime."""

import hashlib
import json
import os
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_gym.dataset_staging import stage_file


SOURCE_REVISION = "e32a2980e72fe6eb04ee04eb749458f570625663"
AUDIT_SHA256 = "fe11495a39976e6aa7d2fb5bea7ee45bf184066ff1a6eec7a58ab56cf2e3189f"
RUNTIME_SHA256 = "8e1cafa1b5b97a52f2a1c2cef63a3e2a54c4c6f5fe3855b7b7195895f8d43496"
TASK = "capsule-9070543"
CREATOR_IMAGE = "registry.codeocean.com/published/4dfd5a01-8d79-40ac-9d7a-10915b8b0e2e@sha256:b0f02683bde424099bb568607b13175d7b0c2603fb0b278e3f9926d9275615b5"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(prepared, audit_path, runtime):
    if digest(audit_path) != AUDIT_SHA256 or digest(runtime) != RUNTIME_SHA256:
        raise ValueError("CORE-Bench source audit or runtime digest mismatch")
    audit = json.loads(audit_path.read_text())
    if audit["creator_revision"] != SOURCE_REVISION or audit["frozen_task"] != TASK:
        raise ValueError("Unexpected scientific capsule/revision")
    root = prepared / "tasks" / TASK
    expected = audit["prepared_files"]
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != set(expected):
        raise ValueError("Unrecorded or missing prepared task files")
    for relative, checksum in expected.items():
        p = root / relative
        if p.is_symlink() or digest(p) != checksum:
            raise ValueError(f"Changed prepared source: {relative}")
    environment_files = {p for p in expected if p.startswith("environment/")}
    if any("/results/" in p or p.endswith("/answers.json") or "/tests/" in p for p in environment_files):
        raise ValueError("Reference/scoring assets leaked into solver build context")
    return audit


def stage(prepared, audit_path, runtime, destination):
    audit = validate(prepared, audit_path, runtime)
    with TemporaryDirectory() as temp:
        temp = Path(temp)
        with tarfile.open(runtime, "r:gz") as archive:
            members = archive.getmembers()
            for item in members:
                relative = Path(item.name)
                if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "python":
                    raise ValueError("Unsafe runtime archive path")
                if not (item.isfile() or item.isdir() or item.issym()):
                    raise ValueError("Unsupported runtime archive member")
            archive.extractall(temp, filter="data")
        target = destination / "tasks" / TASK
        for relative, checksum in audit["prepared_files"].items():
            src = prepared / "tasks" / TASK / relative
            if relative == "task.toml":
                overlay = temp / "task.toml"
                overlay.write_text('artifacts = ["/workspace"]\n' + src.read_text())
                stage_file(overlay, target / relative)
            elif relative == "environment/Dockerfile":
                overlay = temp / "Dockerfile"
                content = src.read_text().splitlines()
                content[0] = "FROM " + CREATOR_IMAGE
                content.append(
                    "RUN mkdir -p /opt/corebench-build-metadata && dpkg-query -W > /opt/corebench-build-metadata/dpkg.tsv && python3 -m pip freeze > /opt/corebench-build-metadata/pip.txt"
                )
                overlay.write_text("\n".join(content) + "\n")
                stage_file(overlay, target / relative)
            else:
                stage_file(src, target / relative, expected_sha256=checksum)
                (target / relative).chmod(src.stat().st_mode & 0o777)
        for src in sorted((temp / "python").rglob("*")):
            dst = target / "environment" / src.relative_to(temp)
            if src.is_dir():
                continue
            if src.is_symlink():
                # The verified archive's data filter enforces links inside its root.
                dst.parent.mkdir(parents=True, exist_ok=True)
                link = os.readlink(src)
                if dst.is_symlink() and os.readlink(dst) == link:
                    continue
                if dst.exists() or dst.is_symlink():
                    raise ValueError("Existing runtime link differs")
                dst.symlink_to(link)
            else:
                stage_file(src, dst, expected_sha256=digest(src))
                dst.chmod(src.stat().st_mode & 0o777)
        row = {
            "instance_id": f"corebench::{TASK}",
            "task_id": f"corebench::{TASK}",
            "responses_create_params": {"input": [], "temperature": 0.0},
            "agent_ref": {"name": "harbor_agent"},
        }
        dataset = temp / "benchmark.jsonl"
        dataset.write_text(json.dumps(row) + "\n")
        receipt = temp / "provenance.json"
        receipt.write_text(
            json.dumps(
                {
                    "source_revision": SOURCE_REVISION,
                    "audit_sha256": AUDIT_SHA256,
                    "runtime_sha256": RUNTIME_SHA256,
                    "creator_image": CREATOR_IMAGE,
                    "build_packages": "Resolved apt/transitive pip versions recorded inside image; final image digest required before live",
                    "task": TASK,
                    "regime": "medium",
                    "rows": 1,
                    "dataset_sha256": digest(dataset),
                    "coverage": "One frozen capsule; not full CORE-Bench",
                },
                indent=2,
            )
            + "\n"
        )
        stage_file(receipt, destination / "provenance.json")
        stage_file(dataset, destination / "benchmark.jsonl")
    return destination / "benchmark.jsonl"


def prepare():
    return stage(
        Path(os.environ["COREBENCH_PREPARED_DIR"]),
        Path(os.environ["COREBENCH_SOURCE_AUDIT"]),
        Path(os.environ["COREBENCH_RUNTIME_ARCHIVE"]),
        Path(__file__).parent / "data",
    )


if __name__ == "__main__":
    print(prepare())
