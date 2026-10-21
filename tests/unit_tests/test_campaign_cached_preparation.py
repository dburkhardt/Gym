# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the native preparation cache, with real campaign staging and receipts."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import benchmarks.fgbench.prepare as adapter
from nemo_gym.cli.eval import PrepareBenchmarkConfig, _prepare_datasets


def write_source(path, identity):
    path.write_text(
        json.dumps({"uuid": identity, "responses_create_params": {"input": "question"}, "verifier_metadata": {}})
        + "\n"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    monkeypatch.setenv("CAMPAIGN_SAMPLE_JSONL", str(source))
    monkeypatch.setenv("CAMPAIGN_SAMPLE_SHA256", write_source(source, "first"))
    monkeypatch.setattr(adapter, "__file__", str(tmp_path / "prepare.py"))
    target = adapter.prepare()

    def run(path=target):
        dataset = SimpleNamespace(
            prepare_script="benchmarks/fgbench/prepare.py", jsonl_fpath=str(path), prepare_dependencies=[]
        )
        _prepare_datasets(
            [SimpleNamespace(name="fgbench", dataset=dataset)],
            PrepareBenchmarkConfig(use_cached_prepared_benchmarks=True, num_prepare_benchmark_processes=1),
        )

    return source, target, run


def test_valid_native_cache_untouched(campaign):
    _, target, run = campaign
    original = (target.read_bytes(), target.stat().st_mtime_ns, target.with_suffix(".source.json").read_bytes())
    run()
    assert original == (
        target.read_bytes(),
        target.stat().st_mtime_ns,
        target.with_suffix(".source.json").read_bytes(),
    )


@pytest.mark.parametrize(
    "damage", ["changed_source", "altered_output", "missing_receipt", "partial_receipt", "wrong_ids"]
)
def test_native_cache_repairs_only_from_selected_source(campaign, monkeypatch, damage):
    source, target, run = campaign
    receipt = target.with_suffix(".source.json")
    if damage == "changed_source":
        monkeypatch.setenv("CAMPAIGN_SAMPLE_SHA256", write_source(source, "second"))
    elif damage == "altered_output":
        target.write_text(target.read_text().replace("question", "altered"))
    elif damage == "missing_receipt":
        receipt.unlink()
    elif damage == "partial_receipt":
        receipt.write_text('{"source_sha256":')
    else:
        data = json.loads(receipt.read_text())
        data["task_ids"] = ["incorrect"]
        receipt.write_text(json.dumps(data))
    run()
    row = json.loads(target.read_text())
    assert row["task_id"] == ("second" if damage == "changed_source" else "first")
    assert row["responses_create_params"]["input"] == "question"
    proof = json.loads(receipt.read_text())
    assert proof["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert proof["output_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()


def test_source_pin_mismatch_preserves_cache(campaign, monkeypatch):
    _, target, run = campaign
    before = target.read_bytes()
    monkeypatch.setenv("CAMPAIGN_SAMPLE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run()
    assert target.read_bytes() == before


def test_external_prepared_override_is_not_rewritten(campaign, tmp_path, monkeypatch):
    source, target, run = campaign
    external = tmp_path / "external.jsonl"
    external.write_bytes(source.read_bytes())
    before = target.read_bytes()
    run(external)
    external.write_text("external user data")
    with pytest.raises(ValueError, match="External prepared sample"):
        run(external)
    assert external.read_text() == "external user data"
    assert target.read_bytes() == before
    monkeypatch.delenv("CAMPAIGN_SAMPLE_JSONL")
    monkeypatch.delenv("CAMPAIGN_SAMPLE_SHA256")
    run(external)  # Existing explicit prepared-input workflow, not a campaign-source claim.
    assert external.read_text() == "external user data"


def test_unrelated_module_without_receipt_keeps_native_behavior(campaign, monkeypatch):
    import nemo_gym.cli.eval as cli

    _, target, run = campaign
    target.with_suffix(".source.json").unlink()

    def forbidden():
        raise AssertionError("Unrelated cached preparation should not run")

    monkeypatch.setattr(cli.importlib, "import_module", lambda _: SimpleNamespace(prepare=forbidden))
    run()


@pytest.mark.parametrize("alias", ["dotdot", "parent_symlink"])
def test_canonical_alias_cannot_bypass_missing_receipt(campaign, tmp_path, monkeypatch, alias):
    from benchmarks.scientific_agent_campaign.stage_sample import is_sample_current

    _, target, _ = campaign
    monkeypatch.delenv("CAMPAIGN_SAMPLE_JSONL")
    monkeypatch.delenv("CAMPAIGN_SAMPLE_SHA256")
    target.with_suffix(".source.json").unlink()
    if alias == "dotdot":
        child = target.parent / "child"
        child.mkdir()
        alternative = child / ".." / target.name
    else:
        link = tmp_path / "data-link"
        link.symlink_to(target.parent, target_is_directory=True)
        alternative = link / target.name
    assert not is_sample_current(alternative, default_destination=target)


def test_changed_normalization_contract_invalidates_cache(campaign):
    from benchmarks.scientific_agent_campaign.stage_sample import is_sample_current

    _, target, _ = campaign
    assert not is_sample_current(target, default_destination=target, expected_normalizations=["image_detail_auto"])


def test_stale_receipt_backed_external_preserved(campaign, tmp_path, monkeypatch):
    from benchmarks.scientific_agent_campaign.stage_sample import prepare_sample

    source, target, run = campaign
    external = prepare_sample(tmp_path / "external.jsonl", source=source)
    original = external.read_bytes(), external.with_suffix(".source.json").read_bytes(), target.read_bytes()
    monkeypatch.setenv("CAMPAIGN_SAMPLE_SHA256", write_source(source, "second"))
    with pytest.raises(ValueError, match="provenance is invalid or stale"):
        run(external)
    assert original == (external.read_bytes(), external.with_suffix(".source.json").read_bytes(), target.read_bytes())


def test_genehop_explicit_source_selection(tmp_path, monkeypatch):
    import benchmarks.genehop.prepare as genehop

    source = tmp_path / "genehop.jsonl"
    target = tmp_path / "prepared.jsonl"

    def write(identity):
        source.write_text(
            json.dumps(
                {
                    "uuid": identity,
                    "responses_create_params": {"input": [{"role": "user", "content": "question"}]},
                    "verifier_metadata": {"question": "question"},
                }
            )
            + "\n"
        )

    write("first")
    monkeypatch.delenv("CAMPAIGN_SAMPLE_JSONL", raising=False)
    monkeypatch.delenv("CAMPAIGN_SAMPLE_SHA256", raising=False)
    monkeypatch.setenv("GENEHOP_SAMPLE", str(source))
    monkeypatch.setenv("GENEHOP_PREPARED_SAMPLE", str(target))
    genehop.prepare()
    assert genehop.is_prepared_data_current(target)
    write("second")
    assert not genehop.is_prepared_data_current(target)


def test_all_shared_sample_consumers_expose_native_hook():
    import ast

    for file in Path("benchmarks").glob("*/prepare.py"):
        text = file.read_text()
        if "from benchmarks.scientific_agent_campaign.stage_sample import prepare_sample" in text:
            functions = {node.name for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
            assert "is_prepared_data_current" in functions, file


def test_fukuyama_alternate_source_and_normalization(tmp_path, monkeypatch):
    import benchmarks.fukuyamabench.prepare as fukuyama

    source = tmp_path / "frozen.jsonl"
    target = tmp_path / "output.jsonl"
    row = {
        "uuid": "A041",
        "stratum": "A",
        "responses_create_params": {"input": "question"},
        "verifier_metadata": {"pathway": [{"products": ["C"]}], "checkpoints": [[1]]},
    }
    source.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(fukuyama, "OUTPUT_FPATH", target)
    monkeypatch.setenv("FUKUYAMA_FROZEN_SAMPLE", str(source))
    monkeypatch.setenv("CAMPAIGN_SAMPLE_JSONL", str(tmp_path / "not-used.jsonl"))
    monkeypatch.setenv("CAMPAIGN_SAMPLE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    fukuyama.prepare()
    assert fukuyama.is_prepared_data_current(target)
    receipt = target.with_suffix(".source.json")
    proof = json.loads(receipt.read_text())
    assert proof["normalizations"] == ["fukuyama_native_metadata_aliases"]
    proof["normalizations"] = []
    receipt.write_text(json.dumps(proof))
    assert not fukuyama.is_prepared_data_current(target)
    monkeypatch.setenv("CAMPAIGN_SAMPLE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fukuyama.is_prepared_data_current(target)
