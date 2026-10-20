# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load(ROOT / "benchmarks/genotex/prepare.py", "genotex_prepare")
grade_module = load(ROOT / "responses_api_agents/genotex/grade.py", "genotex_grade")
runtime = load(ROOT / "responses_api_agents/genotex/runtime.py", "genotex_runtime")


def test_selection_is_deterministic_and_requires_full_stage_reference():
    metadata = {"A": {"conditions": ["Age", "Gender", "B"]}, "B": {"conditions": []}, "C": {"conditions": []}}
    entries = [
        {"path": f"input/GEO/{t}/GSE1/raw.gz", "type": "file", "size": n} for t, n in [("A", 20), ("B", 10), ("C", 1)]
    ]
    entries += [
        {"path": f"output/regress/{t}/significant_genes_condition_{c}.json", "type": "file", "size": 1}
        for t, c in [("A", None), ("A", "Age"), ("A", "Gender"), ("A", "B"), ("B", None)]
    ]
    chosen, excluded = prepare.select_pairs(metadata, entries)
    assert [x["task_id"] for x in chosen] == ["B__None", "A__None", "A__Age", "A__Gender", "A__B"]
    assert "No published regression" in next(x for x in excluded if x["trait"] == "C")["reason"]
    assert prepare.select_pairs(metadata, list(reversed(entries)))[0] == chosen


@pytest.fixture
def fixture(tmp_path):
    pytest.importorskip("sklearn", reason="Use the dedicated GenoTEX grading environment")
    pytest.importorskip("sparse_lmm", reason="Use the dedicated GenoTEX grading environment")
    upstream = Path(os.environ.get("GENOTEX_UPSTREAM", "/tmp/genotex-assets/upstream"))
    if not (upstream / "eval.py").exists():
        pytest.skip("Set GENOTEX_UPSTREAM to the pinned upstream source checkout")
    ref, pred, staging = [tmp_path / name for name in ["reference", "prediction", "grading"]]
    for d in [ref, pred, staging / "metadata"]:
        d.mkdir(parents=True)
    (staging / "metadata/task_info.json").write_text(
        json.dumps({"Trait": {"conditions": [], "related_genes": ["A", "B"]}})
    )
    cohort = ref / "preprocess/Trait"
    cohort.mkdir(parents=True)
    (cohort / "cohort_info.json").write_text(
        json.dumps(
            {"GSE1": {"is_available": True, "is_usable": True, "has_age": True, "has_gender": True, "sample_size": 3}}
        )
    )
    (cohort / "GSE1.csv").write_text(",Trait,A,B\ns1,0,1,5\ns2,1,2,6\ns3,1,3,7\n")
    for name in ["clinical_data", "gene_data"]:
        (cohort / name).mkdir()
        (cohort / name / "GSE1.csv").write_text(",s1,s2,s3\nA,1,2,3\nB,5,6,7\n")
    regression = ref / "regress/Trait/significant_genes_condition_None.json"
    regression.parent.mkdir(parents=True)
    regression.write_text(json.dumps({"significant_genes": {"Variable": ["A", "B"]}}))
    shutil.copytree(ref, pred, dirs_exist_ok=True)
    return upstream, staging, pred, ref


def test_upstream_correct_fixture_all_three_stages(fixture):
    result = grade_module.grade(*fixture, "Trait", None)
    assert result["reward"] == 1
    assert result["complete"]
    assert result["raw_upstream"]["selection"]["selection_metrics"]["average"]["accuracy"] == 100
    assert result["raw_upstream"]["preprocessing"]["linked"]["average"]["composite_similarity_correlation"] == 1


def test_wrong_gene_set_is_zero(fixture):
    (fixture[2] / "regress/Trait/significant_genes_condition_None.json").write_text(
        json.dumps({"significant_genes": {"Variable": ["WRONG"]}})
    )
    result = grade_module.grade(*fixture, "Trait", None)
    assert result["reward"] == 0
    assert result["raw_upstream"]["analysis"]["categorized"]["Overall"]["success"] == 100


def test_missing_is_distinct_from_wrong(fixture):
    (fixture[2] / "regress/Trait/significant_genes_condition_None.json").unlink()
    result = grade_module.grade(*fixture, "Trait", None)
    assert result["reward"] == 0
    assert not result["complete"]
    assert result["raw_upstream"]["analysis"]["categorized"]["Overall"]["success"] == 0
    assert result["raw_upstream"]["analysis"]["categorized"]["Overall"]["f1"] is None


def test_malformed_output_preserves_upstream_omission(fixture):
    (fixture[2] / "regress/Trait/significant_genes_condition_None.json").write_text("not json")
    result = grade_module.grade(*fixture, "Trait", None)
    assert result["reward"] == 0
    assert result["raw_upstream"]["analysis"]["categorized"]["Overall"] == {}


def test_reject_solver_output_symlinks(fixture):
    (fixture[2] / "escape").symlink_to(fixture[3])
    with pytest.raises(ValueError, match="symlink"):
        grade_module.grade(*fixture, "Trait", None)


def test_reference_isolation(tmp_path):
    root = tmp_path / "data"
    for p in [
        "upstream/tools",
        "assets/metadata",
        "assets/input/GEO/Trait",
        "assets/output/preprocess/Trait",
        "assets/output/regress/Trait",
    ]:
        (root / p).mkdir(parents=True)
    (root / "upstream/tools/helper.py").write_text("pass")
    (root / "upstream/eval.py").write_text("SECRET-GRADER")
    (root / "assets/metadata/task_info.json").write_text("{}")
    (root / "assets/input/GEO/Trait/raw.txt").write_text("raw")
    (root / "assets/output/preprocess/Trait/answer.csv").write_text("SECRET-ANSWER")
    (root / "assets/output/regress/Trait/significant_genes_condition_None.json").write_text("SECRET-GENES")
    paths = runtime.stage(root, tmp_path / "run", {"trait": "Trait", "condition": None, "traits": ["Trait"]})
    public_text = "".join(p.read_text() for p in paths["public"].rglob("*") if p.is_file())
    assert "SECRET" not in public_text
    assert not (paths["public"] / "output").exists()
    assert not (paths["public"] / "eval.py").exists()
    assert (paths["reference"] / "regress/Trait/significant_genes_condition_None.json").exists()


def test_bad_trait_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsafe"):
        runtime.stage(tmp_path, tmp_path / "run", {"trait": "../escape", "condition": None, "traits": ["../escape"]})


def test_reference_code_is_not_a_required_solver_artifact(fixture):
    code = fixture[3] / "preprocess/Trait/code"
    code.mkdir()
    (code / "TCGA.py").write_text("pass")
    result = grade_module.grade(*fixture, "Trait", None)
    assert result["complete"]
    assert result["expected_files"] == 5
    assert result["auxiliary_reference_files"] == ["preprocess/Trait/code/TCGA.py"]


def test_expansion_retains_original_ids_and_fixed_strata():
    metadata = {f"T{i}": {"conditions": ["Age", "Gender", "T0"]} for i in range(30)}
    entries = [{"path": f"input/GEO/{t}/G/raw.gz", "type": "file", "size": i+1} for i,t in enumerate(metadata)]
    entries += [{"path": f"output/regress/{t}/significant_genes_condition_{c}.json", "type":"file", "size":1} for t in metadata for c in [None,"Age","Gender","T0"]]
    five,_=prepare.select_pairs(metadata,entries)
    twenty,_=prepare.select_pairs(metadata,entries,20)
    fifty,_=prepare.select_pairs(metadata,entries,50)
    assert len(twenty)==20 and len(fifty)==50
    assert {r["task_id"] for r in five} <= {r["task_id"] for r in twenty} <= {r["task_id"] for r in fifty}
    assert [sum(r["category"]==c for r in twenty) for c in ["unconditional","conditional","two-step"]]==[8,8,4]
    assert prepare.select_pairs(metadata,list(reversed(entries)),20)[0]==twenty
    with pytest.raises(ValueError):prepare.select_pairs(metadata,entries,19)
