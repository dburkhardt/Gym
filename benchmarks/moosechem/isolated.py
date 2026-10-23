# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public solver inputs and separate unchanged creator verification.

Filesystem exclusion is enforced by the caller's sandbox mounts, not this module.
"""

import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Callable

from benchmarks.moosechem.native import MODEL, claim_run, creator_defaults, run


PUBLIC_FILES = (
    "Method/inspiration_screening.py",
    "Method/hypothesis_generation.py",
    "Method/utils.py",
    "Method/logging_utils.py",
    "Data/Inspiration_Corpus_150.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pins() -> dict[str, str]:
    return json.loads(Path(__file__).with_name("source-sha256.json").read_text())


def verify_creator(repo: Path) -> None:
    for name, expected in pins().items():
        if digest(repo / name) != expected:
            raise ValueError(f"Creator file changed: {name}")


def export_public(repo: Path, output: Path, task_id: int) -> dict:
    """Use the creator loader once on the trusted side, exposing only public fields."""
    verify_creator(repo)
    sys.path.insert(0, str(repo))
    utils = importlib.import_module("Method.utils")
    if Path(utils.__file__).resolve() != (repo / "Method/utils.py").resolve():
        raise ValueError("Creator import interception")
    data = utils.load_chem_annotation(str(repo / "Data/chem_research_2024.xlsx"), 1, 1)
    if type(task_id) is not int or not 0 <= task_id < len(data[0]):
        raise ValueError("Invalid creator task ID")
    question = data[0][task_id]
    task = {"original_task_id": task_id, "internal_task_id": 0, "question": question, "survey": data[2][question]}
    output.mkdir(parents=True, exist_ok=False)
    for name in PUBLIC_FILES:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, destination)
    (output / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2))
    manifest = {name: digest(output / name) for name in (*PUBLIC_FILES, "task.json")}
    (output / "public-manifest.json").write_text(json.dumps(manifest, indent=2))
    return task


def validate_public(repo: Path, task_path: Path, *, expected_manifest_sha256: str) -> dict:
    if digest(repo / "public-manifest.json") != expected_manifest_sha256:
        raise ValueError("Trusted public manifest binding mismatch")
    expected_names = {*PUBLIC_FILES, "task.json", "public-manifest.json"}
    actual = set()
    for path in repo.rglob("*"):
        if path.is_symlink():
            raise ValueError("Public bundle symlink forbidden")
        if path.is_file():
            actual.add(str(path.relative_to(repo)))
    if actual != expected_names or task_path.resolve() != (repo / "task.json").resolve():
        raise ValueError("Public input allowlist mismatch")
    manifest = json.loads((repo / "public-manifest.json").read_text())
    if set(manifest) != expected_names - {"public-manifest.json"}:
        raise ValueError("Public manifest key mismatch")
    for name, expected in manifest.items():
        if digest(repo / name) != expected or (name in PUBLIC_FILES and expected != pins()[name]):
            raise ValueError(f"Public input changed: {name}")
    task = json.loads(task_path.read_text())
    if set(task) != {"original_task_id", "internal_task_id", "question", "survey"}:
        raise ValueError("Unexpected public task fields")
    if (
        type(task["original_task_id"]) is not int
        or task["original_task_id"] < 0
        or type(task["internal_task_id"]) is not int
        or task["internal_task_id"] != 0
    ):
        raise ValueError("Invalid public task mapping")
    if not all(isinstance(task[k], str) and task[k] for k in ("question", "survey")):
        raise ValueError("Invalid public text")
    return task


def run_solver(
    repo: Path,
    output: Path,
    task_path: Path,
    *,
    client_factory: Callable[..., object],
    output_token_budget: int,
    expected_manifest_sha256: str,
) -> dict:
    task = validate_public(repo, task_path, expected_manifest_sha256=expected_manifest_sha256)
    return run(
        repo,
        output,
        task["original_task_id"],
        public_task=task,
        client_factory=client_factory,
        output_token_budget=output_token_budget,
    )


def run_verifier(
    repo: Path,
    solver_output: Path,
    output: Path,
    task_id: int,
    *,
    client_factory: Callable[..., object],
    output_token_budget: int,
) -> dict:
    """Run creator recall and reference judge outside the solver; no regeneration."""
    verify_creator(repo)
    output.mkdir(parents=True, exist_ok=False)
    claim_run(output)
    sys.path.insert(0, str(repo))
    screen_module = importlib.import_module("Method.inspiration_screening")
    evaluator_module = importlib.import_module("Method.evaluate")
    for module, stage in ((screen_module, "inspiration_screening"), (evaluator_module, "evaluate")):
        if Path(module.__file__).resolve() != (repo / f"Method/{stage}.py").resolve():
            raise ValueError("Creator import interception")
        module.OpenAI = lambda _stage=stage, **kw: client_factory(
            stage=_stage, trace=output / "model-traces.jsonl", replay=None, output_token_budget=output_token_budget
        )
    for name in ("utils", "logging_utils"):
        module = importlib.import_module("Method." + name)
        if Path(module.__file__).resolve() != (repo / "Method" / f"{name}.py").resolve():
            raise ValueError("Creator import interception")
    args = [creator_defaults(repo / "Method" / f"{name}.py") for name in ("inspiration_screening", "evaluate")]
    for arg in args:
        vars(arg).update(
            api_type=0,
            api_key="gym-task-socket",
            base_url="http://127.0.0.1:49180/v1",
            model_name=MODEL,
            chem_annotation_path=str(repo / "Data/chem_research_2024.xlsx"),
            custom_inspiration_corpus_path=str(repo / "Data/Inspiration_Corpus_150.json"),
            custom_research_background_path="",
            corpus_size=150,
            background_question_id=task_id,
            if_save=1,
            if_use_strict_survey_question=1,
            if_use_background_survey=1,
        )
    screening = screen_module.Screening(args[0])
    question = screening.bkg_q_list[task_id]
    raw = [json.loads(line) for line in (solver_output / "raw-screening.jsonl").read_text().splitlines()]
    if len(raw) != 4 or any(set(row) != {"question", "windows"} or row["question"] != question for row in raw):
        raise ValueError("Missing or mismatched four creator screening rounds")
    public_screen = json.loads((solver_output / "screening.json").read_text())
    utils = importlib.import_module("Method.utils")
    organized = utils.organize_raw_inspirations({question: [r["windows"] for r in raw]})
    if public_screen != [organized, {}]:
        raise ValueError("Raw screening windows differ from saved public selection")
    ratios = [screening.check_how_many_hit_groundtruth_insp(question, row["windows"]) for row in raw]
    (output / "screening.json").write_text(json.dumps([organized, {question: ratios}]))
    shutil.copyfile(solver_output / "hypotheses.json", output / "hypotheses.json")
    vars(args[1]).update(
        hypothesis_dir=str(output / "hypotheses.json"),
        output_dir=str(output / "evaluation.json"),
        if_load_from_saved=0,
        if_with_gdth_hyp_annotation=1,
    )
    evaluator = evaluator_module.Evaluate(args[1])
    evaluator.run()
    result = {
        "scope": "creator_separate_reference_verifier",
        "original_task_id": task_id,
        "evaluation": json.loads((output / "evaluation.json").read_text()),
        "reward": ratios[-1][1],
        "reward_metric": "final_round_inspiration_top3_recall_not_hypothesis_quality",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    return result
