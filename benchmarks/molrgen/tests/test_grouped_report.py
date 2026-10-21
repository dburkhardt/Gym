# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
from pathlib import Path

import pytest

from benchmarks.molrgen.grouped_report import creator_functions, report


@pytest.fixture
def source():
    path = Path(__file__).resolve().parents[4] / "molrgen-upstream"
    if not path.exists():
        pytest.skip("Pinned MolRGen creator checkout unavailable")
    return path


def row(index, smiles, reward):
    return {
        "_ng_rollout_index": index,
        "reward": reward,
        "mask_sample": False,
        "responses_create_params": {"input": [{"role": "user", "content": "same objective"}]},
        "response": {"model": "fixture", "status": "completed"},
        "verifier_metadata": {"metadata": {"prompt_id": "p"}},
        "diagnostics": {
            "creator_response": {
                "reward": reward,
                "request": {"query": [""], "metadata": [{"prompt_id": "p"}]},
                "config": {"oracle": "fixture"},
                "oracle_variant": "fixture",
                "creator_response": {
                    "rewards": [reward],
                    "verifier_metadatas": [
                        {
                            "generation_verifier_metadata": {
                                "all_smi": [smiles] if smiles else [],
                                "all_smi_rewards": [reward],
                            }
                        }
                    ],
                },
            }
        },
    }


def call(rows, source, **kwargs):
    return report(
        rows,
        expected_draws={"p": len(rows)},
        source_root=source,
        k=kwargs.get("k", 2),
        similarity_threshold=0.5,
        fingerprint="ecfp4-128",
    )


def test_creator_canonical_first_duplicate_and_padding(source):
    functions = creator_functions(source)
    # OCC and CCO canonicalize identically; first0.2 survives, not later0.9.
    assert functions["top_k"](["OCC", "CCO", "N"], [0.2, 0.9, 0.4], 4) == pytest.approx(0.15)
    result = call([row(0, "OCC", 0.2), row(1, "CCO", 0.9), row(2, "N", 0.4)], source, k=4)
    assert result["groups"][0]["top_k"] == pytest.approx(0.15)
    assert result["published_sampling_reproduced"] is False


def test_exact_creator_diversity_fixture(source):
    molecules = ["CCC", "CC", "CC(C)C", "CCNC", "CCNC"]
    scores = [0.9, 0.6, 0.8, 1.0, 1.0]
    result = call([row(i, m, s) for i, (m, s) in enumerate(zip(molecules, scores))], source)
    assert result["groups"][0]["diversity_aware_top_k"] == pytest.approx(0.95)
    assert result["groups"][0]["raw_scores"] == scores


def test_singleton_missing_masked_and_budget_zero(source):
    one = row(0, "CC", 0.0)
    result = report(
        [one], expected_draws={"p": 15}, source_root=source, k=10, similarity_threshold=0.7, fingerprint="ecfp4-2048"
    )
    assert result["scored_groups"] == 0 and result["groups"][0]["top_k"] is None
    assert result["observed_draws"] == 1 and result["expected_draws"] == 15
    with pytest.raises(ValueError, match="singleton"):
        call([one], source)
    budget = row(0, "", 0.0)
    budget["failure_kind"] = "model_budget"
    budget["response"]["status"] = "incomplete"
    budget["response"]["incomplete_details"] = {"reason": "max_output_tokens"}
    result = call([budget, row(1, "CC", 0.8)], source)
    assert result["scored_groups"] == 1
    assert result["groups"][0]["top_k"] == 0.4
    broken = copy.deepcopy(budget)
    broken["mask_sample"] = True
    assert call([broken, row(1, "CC", 0.8)], source)["scored_groups"] == 0
    filtered = copy.deepcopy(budget)
    filtered["response"]["incomplete_details"] = {"reason": "content_filter"}
    assert call([filtered, row(1, "CC", 0.8)], source)["scored_groups"] == 0
    assert result["groups"][0]["draws"][0]["failure_kind"] == "model_budget"


def test_duplicate_and_protocol_mixing_refused(source):
    with pytest.raises(ValueError, match="duplicate"):
        call([row(0, "CC", 0.1), row(0, "N", 0.2)], source)
    changed = row(1, "N", 0.2)
    changed["response"]["model"] = "other"
    with pytest.raises(ValueError, match="Mixed"):
        call([row(0, "CC", 0.1), changed], source)


def test_dirty_source_refused(source, tmp_path):
    import shutil

    shutil.copytree(source / "molrgen/evaluation", tmp_path / "molrgen/evaluation")
    (tmp_path / "notebooks/utils").mkdir(parents=True)
    shutil.copy(source / "notebooks/utils/molgen_utils.py", tmp_path / "notebooks/utils/molgen_utils.py")
    (tmp_path / "molrgen/evaluation/top_k.py").write_text("tampered")
    with pytest.raises(ValueError, match="source mismatch"):
        creator_functions(tmp_path)


def test_receipt_pairing_and_last_molecule_convention(source):
    multiple = row(0, "CC", 0.2)
    meta = multiple["diagnostics"]["creator_response"]["creator_response"]["verifier_metadatas"][0][
        "generation_verifier_metadata"
    ]
    meta.update(all_smi=["CC", "N"], all_smi_rewards=[0.2, 0.9])
    result = call([multiple, row(1, "O", 0.4)], source)
    assert result["groups"][0]["raw_scores"] == [0.2, 0.4]
    assert result["groups"][0]["selected_molecular_scores"] == [0.9, 0.4]
    changed = row(0, "CC", 0.2)
    changed["diagnostics"]["creator_response"]["request"]["query"] = ["different response"]
    assert call([changed, row(1, "O", 0.4)], source)["scored_groups"] == 0


def test_cli_never_overwrites_prior_report(source, tmp_path, monkeypatch):
    import json
    import runpy
    import sys

    rows = tmp_path / "rows.jsonl"
    rows.write_text("\n".join(json.dumps(x) for x in [row(0, "CC", 0.2), row(1, "N", 0.4)]))
    scope = tmp_path / "scope.json"
    scope.write_text(
        json.dumps({"expected_draws": {"p": 2}, "k": 2, "similarity_threshold": 0.5, "fingerprint": "ecfp4-128"})
    )
    output = tmp_path / "report.json"
    script = Path(__file__).parents[1] / "grouped_report.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "--rows", str(rows), "--scope", str(scope), "--source", str(source), "--output", str(output)],
    )
    runpy.run_path(str(script), run_name="__main__")
    preserved = output.read_bytes()
    with pytest.raises(FileExistsError):
        runpy.run_path(str(script), run_name="__main__")
    assert output.read_bytes() == preserved
