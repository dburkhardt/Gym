# SPDX-License-Identifier: Apache-2.0
"""Validate pinned BIABench assets and publish native task identity, without inference."""

import hashlib
import importlib
import inspect
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_gym.dataset_staging import stage_file


def validate(root: Path, files: dict[str, str]) -> None:
    root = root.resolve()
    for relative, expected in files.items():
        path = root / relative
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError(f"Unsafe pinned asset: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Pinned BIABench asset mismatch: {relative}")


def stage(source: Path, task: Path, destination: Path) -> Path:
    manifest = json.loads(Path(__file__).with_name("frozen_sources.json").read_text())
    validate(source, manifest["creator_files"])
    validate(task, manifest["task_files"])
    source, task = source.resolve(), task.resolve()
    public = {name: sha for name, sha in manifest["task_files"].items() if name.startswith("input/")}
    solver = destination / "solver"
    for path in solver.rglob("*"):
        if path.is_symlink():
            raise ValueError("Solver staging must not contain symlinks")
        if path.is_file():
            relative = str(path.relative_to(solver))
            if path.name.startswith(".") and path.name.endswith(".stage.lock"):
                relative = str(path.relative_to(solver).with_name(path.name[1:-11]))
            if relative not in public and relative != "instruction.md":
                raise ValueError(f"Unexpected solver file: {relative}")
    for relative, checksum in manifest["task_files"].items():
        role = "solver" if relative in public else "references"
        stage_file(task / relative, destination / role / relative, expected_sha256=checksum)
    for relative, checksum in manifest["creator_files"].items():
        stage_file(source / relative, destination / "verifier_source" / relative, expected_sha256=checksum)
    sys.path.insert(0, str(source))
    native = importlib.import_module("bioimage_agent_bench.task_spec")
    if Path(inspect.getfile(native)).resolve() != source / "bioimage_agent_bench/task_spec.py":
        raise ValueError("Wrong creator task renderer imported")
    prompt = (
        native.render_instruction(
            native.load_task_spec(task),
            agent_id="nemotron_harbor",
            level="basic",
            input_dir=Path("/workspace/input"),
            output_dir=Path("/workspace/output"),
            include_environment=False,
        )
        + "\n\nCompute environment: This sandbox provides 4 CPUs and 8 GiB RAM, with no GPU. Use CPU execution.\n"
    )
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    if prompt_hash != "380a3ac2065c676e69ae005df1c59409e693d7e9bb37e664ca6bc85db7e759c5":
        raise ValueError("Frozen BIABench prompt differs from recorded CPU variant")
    row = {
        "task_id": manifest["task_id"],
        "instance_id": "biabench::" + manifest["task_id"],
        "responses_create_params": {"input": []},
        "verifier_metadata": {
            "tier": manifest["tier"],
            "outcome_scorer": manifest["outcome_scorer"],
            "process_score_required": True,
        },
    }
    # Only solver/ may be mounted into the task; references and creator scorer stay separate.
    with TemporaryDirectory() as directory:
        temporary = Path(directory)
        instruction = temporary / "instruction.md"
        instruction.write_text(prompt)
        stage_file(instruction, solver / instruction.name, expected_sha256=prompt_hash)
        data = temporary / "benchmark.jsonl"
        data.write_text(json.dumps(row) + "\n")
        target = stage_file(data, destination / data.name)
        proof = temporary / "benchmark.source.json"
        proof.write_text(
            json.dumps(manifest | {"output_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}, indent=2) + "\n"
        )
        stage_file(proof, destination / proof.name)
    return target


def prepare() -> Path:
    return stage(
        Path(os.environ["BIABENCH_SOURCE"]),
        Path(os.environ["BIABENCH_TASK"]),
        Path(os.environ.get("BIABENCH_DATA", "benchmarks/biabench/data")),
    )


if __name__ == "__main__":
    print(prepare())
