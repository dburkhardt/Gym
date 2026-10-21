# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.fgbench.projection import project
from benchmarks.scientific_agent_campaign import deterministic
from nemo_gym.result_disposition import summarize_dispositions
from resources_servers.campaign_creator_scoring import app


REPO = Path("../fgbench-upstream").resolve()
SAVED = Path("../../outputs/broad-eval-campaign/fgbench/pilot-20-v2.jsonl")
PROTOCOL = {"model": "saved-model", "temperature": 0.1, "max_output_tokens": 16384}


def row(meta, answer):
    task = {"uuid": "control", "verifier_metadata": meta}
    score = deterministic.score("fgbench", REPO, task, answer)
    return {"reward": score["reward"], "metadata": project(REPO, task, score, protocol=PROTOCOL)}


def test_mixed_strata_missing_zero_and_incorrect_boolean():
    if not REPO.exists():
        pytest.skip("Pinned creator unavailable")
    rows = [row({"type": "single_bool_classification", "answer": "True"}, a) for a in ["True", "False", ""]]
    rows += [row({"type": "single_value_regression", "answer": "2"}, a) for a in ["4", "0", ""]]
    rows += [row({"type": "single_value_regression", "answer": "0"}, "4")]
    report = summarize_dispositions(rows)["per_metric"]
    assert report["fgbench/single_bool/accuracy"]["reviewed_value"] == pytest.approx(1 / 3)
    assert report["fgbench/single_bool/accuracy"]["reviewed_denominator"] == 3
    assert report["fgbench/single_bool/valid_rate"]["reviewed_value"] == pytest.approx(2 / 3)
    numeric = report["fgbench/single_value/rmse"]
    assert numeric["reviewed_value"] == 2
    assert numeric["reviewed_mean"] is None
    assert numeric["reviewed_denominator"] == 1
    assert report["fgbench/single_value/valid_rate"]["reviewed_value"] == 0.25
    empty = summarize_dispositions(rows[-2:])["per_metric"]["fgbench/single_value/rmse"]
    assert empty["reviewed_value"] is None and empty["reviewed_denominator"] == 0


def test_saved20_matches_independent_creator_aggregate():
    if not SAVED.exists() or not REPO.exists():
        pytest.skip("Saved creator evidence unavailable")
    saved = [json.loads(line) for line in SAVED.read_text().splitlines()]
    answers = [
        "".join(
            c["text"]
            for m in r["response"]["output"]
            if m["type"] == "message"
            for c in m["content"]
            if c["type"] == "output_text"
        )
        for r in saved
    ]
    creator = deterministic.module(REPO / "result_ananlysis/analysis.py")
    expected = creator.evaluate([r["verifier_metadata"] for r in saved], answers, "standard")
    projected = [row(r["verifier_metadata"], a) for r, a in zip(saved, answers)]
    for r, p in zip(saved, projected):
        assert r["reward"] == p["reward"]
    metrics = summarize_dispositions(projected)["per_metric"]
    assert len(expected) == 6
    for stratum, e in expected.items():
        name = "accuracy" if stratum.endswith("bool") else "rmse"
        actual = metrics[f"fgbench/{stratum}/{name}"]
        assert (
            actual["reviewed_value"] == pytest.approx(e["acc" if name == "accuracy" else "rmse"])
            if e.get("rmse", 1) is not None
            else actual["reviewed_value"] is None
        )
        assert actual["reviewed_denominator"] == (e["n"] if name == "accuracy" else e["n_valid"])
        validity = metrics[f"fgbench/{stratum}/valid_rate"]
        assert validity["reviewed_denominator"] == e["n"]
        assert validity["reviewed_value"] == pytest.approx(e["valid"])


@pytest.mark.asyncio
async def test_actual_native_verify_keeps_raw_and_projects():
    if not SAVED.exists() or not REPO.exists():
        pytest.skip("Saved creator evidence unavailable")
    saved = json.loads(SAVED.read_text().splitlines()[0])
    body = app.VerifyRequest.model_validate(saved)
    server = SimpleNamespace(
        config=SimpleNamespace(benchmark="fgbench", upstream_repo=str(REPO), scoring_timeout_s=30)
    )
    result = await app.CreatorResourcesServer.verify(server, body)
    assert result.reward == saved["reward"]
    assert result.response == body.response
    assert len(result.metadata["verifier_result"]["rewards"]) == 2
    assert result.metadata["source"] == "fgbench"


def test_dirty_source_refused(tmp_path):
    p = tmp_path / "result_ananlysis"
    p.mkdir()
    (p / "analysis.py").write_text("changed")
    with pytest.raises(ValueError, match="scorer bytes"):
        project(tmp_path, {}, {}, protocol=PROTOCOL)
