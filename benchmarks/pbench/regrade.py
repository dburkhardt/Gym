# SPDX-License-Identifier: Apache-2.0
"""Recompute creator single-run P-Bench metrics from preserved solver artifacts."""

import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def regrade(source: Path, task_id: str, workspace: Path, *, missing_result_is_model_output: bool = False) -> dict:
    manifest = json.loads(Path(__file__).with_name("frozen_sources.json").read_text())
    task = next((t for t in manifest["tasks"] if t["harbor_task"] == task_id), None)
    if task is None:
        raise ValueError("Task is outside frozen P-Bench sample")
    source = source.resolve(strict=True)
    paths = {}
    for relative in ("tests/evaluate.py", "tests/answer_key.json"):
        path = source / "tasks" / task_id / relative
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError("Unsafe verifier source")
        if hashlib.sha256(path.read_bytes()).hexdigest() != task["files"][relative]:
            raise ValueError("Changed pinned P-Bench scorer or reference")
        paths[relative] = path
    workspace = workspace.resolve(strict=True)
    prediction = workspace / "result.json"
    if prediction.is_symlink() or (prediction.exists() and not prediction.is_file()):
        raise ValueError("Saved result must be a regular file")
    if not prediction.exists() and not missing_result_is_model_output:
        raise ValueError("Missing saved artifact is not evidence of a model omission")
    before = prediction.read_bytes() if prediction.exists() else None
    spec = importlib.util.spec_from_file_location("pbench_pinned_creator", paths["tests/evaluate.py"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Copy only the saved response; caller workspace and reference files remain unchanged.
    with TemporaryDirectory() as directory:
        case = Path(directory) / "case"
        case.mkdir()
        if before is not None:
            (case / "result.json").write_bytes(before)
        raw, strict = module.load_run(
            Path(directory), {"case": json.loads(paths["tests/answer_key.json"].read_text())}
        )["case"]
    after = prediction.read_bytes() if prediction.exists() else None
    if after != before:
        raise ValueError("Saved prediction changed during regrade")
    return {
        "task_id": task_id,
        "raw": float(raw),
        "strict": float(strict),
        "prediction_sha256": hashlib.sha256(before).hexdigest() if before is not None else None,
        "scorer_sha256": task["files"]["tests/evaluate.py"],
        "new_model_calls": 0,
        "protocol": "creator single-run scores; not three-repeat pass@1/pass@3",
    }
