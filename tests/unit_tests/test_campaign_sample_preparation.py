# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest

from benchmarks.scientific_agent_campaign.stage_sample import prepare_sample


@pytest.mark.parametrize("identity", [{"uuid": "u"}, {"verifier_metadata": {"task_id": "m"}}, {"task_id": "t"}])
def test_preserves_frozen_fields_and_provenance(tmp_path, identity):
    row = {
        "responses_create_params": {"input": [{"role": "user", "content": "unchanged"}]},
        "verifier_metadata": {"answer": "secret"},
        **identity,
    }
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row) + "\n")
    target = tmp_path / "output.jsonl"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert prepare_sample(target, source=source, expected_source_sha256=digest) == target
    output = json.loads(target.read_text())
    for key, value in row.items():
        assert output[key] == value
    assert output["task_id"] in ("u", "m", "t")
    receipt = json.loads(target.with_suffix(".source.json").read_text())
    assert receipt["source_sha256"] == digest
    assert receipt["output_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    before = target.read_bytes()
    prepare_sample(target, source=source, expected_source_sha256=digest)
    assert target.read_bytes() == before


@pytest.mark.parametrize("case", ["duplicate", "missing", "hash"])
def test_invalid_sample_preserves_existing_destination(tmp_path, case):
    row = {"uuid": "u", "responses_create_params": {"input": "question"}, "verifier_metadata": {}}
    if case == "missing":
        del row["uuid"]
    source = tmp_path / "source.jsonl"
    source.write_text((json.dumps(row) + "\n") * (2 if case == "duplicate" else 1))
    target = tmp_path / "output.jsonl"
    target.write_text("prior-valid-output")
    with pytest.raises(ValueError):
        prepare_sample(target, source=source, expected_source_sha256="0" * 64 if case == "hash" else None)
    assert target.read_text() == "prior-valid-output"


def test_explicit_metadata_identity_preserves_source(tmp_path):
    source = tmp_path / "source.jsonl"
    row = {
        "responses_create_params": {"input": [{"role": "user", "content": "unchanged"}]},
        "verifier_metadata": {"id": "246210390723", "metadata": {"prompt_id": "246210390723"}},
    }
    source.write_text(json.dumps(row) + "\n")
    output = prepare_sample(tmp_path / "output.jsonl", source=source, identity_metadata_key="id")
    staged = json.loads(output.read_text())
    assert staged.pop("task_id") == "246210390723"
    assert staged == row


def test_destination_symlink_cannot_redirect_publication(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"uuid": "u", "responses_create_params": {"input": "question"}, "verifier_metadata": {}}) + "\n"
    )
    unrelated = tmp_path / "unrelated.jsonl"
    unrelated.write_text("preserve unrelated artifact")
    destination = tmp_path / "output.jsonl"
    destination.symlink_to(unrelated)
    with pytest.raises(ValueError, match="symlink"):
        prepare_sample(destination, source=source)
    assert destination.is_symlink()
    assert unrelated.read_text() == "preserve unrelated artifact"
    assert not destination.with_suffix(".source.json").exists()


def test_image_normalization_preserves_explicit_detail_and_source(tmp_path):
    import json

    from benchmarks.scientific_agent_campaign.stage_sample import prepare_sample

    row = {
        "verifier_metadata": {"id": 47, "answer": "3"},
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "https://example.org/a.png"},
                        {"type": "input_image", "image_url": "https://example.org/b.png", "detail": "high"},
                    ],
                }
            ]
        },
    }
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row) + "\n")
    original = source.read_bytes()
    output = prepare_sample(
        tmp_path / "output.jsonl", source=source, identity_metadata_key="id", normalize_image_detail=True
    )
    staged = json.loads(output.read_text())
    assert staged["task_id"] == "47"
    assert staged["responses_create_params"]["input"][0]["content"][0]["detail"] == "auto"
    assert staged["responses_create_params"]["input"][0]["content"][1]["detail"] == "high"
    assert source.read_bytes() == original
    assert json.loads(output.with_suffix(".source.json").read_text())["normalizations"] == ["image_detail_auto"]
