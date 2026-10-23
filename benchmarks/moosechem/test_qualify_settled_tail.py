# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.moosechem.native import MODEL
from benchmarks.moosechem.qualify_replay import ReplayRefused
from benchmarks.moosechem.qualify_settled_tail import SettledClient, SettledSequence, TailReached


def fixture(tmp_path):
    source = json.loads(Path(__file__).with_name("source-sha256.json").read_text())
    request = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "one"}],
        "max_tokens": 16384,
        "temperature": 1.0,
    }
    response = {
        "id": "response",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "answer"}}],
    }
    rows = [
        {"stage": "generate", "request": request, "response": response, "started": 1, "elapsed_seconds": 3},
        {
            "stage": "generate",
            "request": {**request, "messages": [{"role": "user", "content": "two"}]},
            "error_type": "InternalServerError",
            "started": 5,
            "elapsed_seconds": 3,
        },
    ]
    saved = tmp_path / "saved"
    saved.write_text("\n".join(map(json.dumps, rows)))
    ledger = []
    for i, row in enumerate(rows):
        wire = json.dumps(row["request"], ensure_ascii=False, separators=(",", ":")).encode()
        ledger += [
            {
                "event": "started",
                "request_id": str(i),
                "time": row["started"] + 1,
                "request_sha256": hashlib.sha256(wire).hexdigest(),
            },
            {
                "event": "finished",
                "request_id": str(i),
                "time": row["started"] + 2,
                "http_status": 200 if i == 0 else 503,
                "provider_response_id": "response" if i == 0 else None,
            },
        ]
    gateway = tmp_path / "gateway"
    gateway.write_text("\n".join(map(json.dumps, ledger)))
    proof = {
        "saved_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
        "gateway_sha256": hashlib.sha256(gateway.read_bytes()).hexdigest(),
        "creator_sha256": source,
        "tail_gateway_request_id": "1",
    }
    evidence = tmp_path / "proof"
    evidence.write_text(json.dumps(proof))
    return saved, gateway, evidence, rows


def sequence(tmp_path, monkeypatch):
    saved, gateway, evidence, rows = fixture(tmp_path)
    from benchmarks.moosechem import qualify_settled_tail as module

    original = module.sha
    sources = json.loads(Path(__file__).with_name("source-sha256.json").read_text())
    monkeypatch.setattr(
        module,
        "sha",
        lambda p: (
            sources.get(str(Path(p).relative_to(tmp_path)), original(p))
            if Path(p).is_file()
            else sources[str(Path(p).relative_to(tmp_path))]
        ),
    )
    return SettledSequence(saved, evidence, gateway, tmp_path), rows


def test_prefix_then_exact_tail_never_falls_through(tmp_path, monkeypatch):
    seq, rows = sequence(tmp_path, monkeypatch)
    client = SettledClient(stage="generate", trace=tmp_path / "trace", replay=seq, output_token_budget=16384)
    assert client.create(**{**rows[0]["request"], "max_tokens": 8192}).id == "response"
    with pytest.raises(TailReached):
        client.create(**{**rows[1]["request"], "max_tokens": 8192})
    assert seq.index == 1
    with pytest.raises(ReplayRefused, match="divergence"):
        client.create(**{**rows[1]["request"], "max_tokens": 8192, "temperature": 0.1})


@pytest.mark.parametrize("field", ["saved_sha256", "gateway_sha256", "creator_sha256"])
def test_missing_or_tampered_provenance_refuses(tmp_path, field):
    saved, gateway, evidence, _ = fixture(tmp_path)
    p = json.loads(evidence.read_text())
    p[field] = {}
    evidence.write_text(json.dumps(p))
    with pytest.raises(ReplayRefused):
        SettledSequence(saved, evidence, gateway, tmp_path)


def test_corrupt_response_refuses_even_rebound_file(tmp_path, monkeypatch):
    seq, rows = sequence(tmp_path, monkeypatch)
    saved = tmp_path / "saved"
    rows[0]["response"]["choices"][0]["finish_reason"] = "length"
    saved.write_text("\n".join(map(json.dumps, rows)))
    evidence = tmp_path / "proof"
    p = json.loads(evidence.read_text())
    p["saved_sha256"] = hashlib.sha256(saved.read_bytes()).hexdigest()
    evidence.write_text(json.dumps(p))
    with pytest.raises(ReplayRefused, match="Corrupt"):
        SettledSequence(saved, evidence, tmp_path / "gateway", tmp_path)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "timeout", "wrong_status"])
def test_unsettled_gateway_refuses(tmp_path, monkeypatch, mutation):
    sequence(tmp_path, monkeypatch)
    gateway = tmp_path / "gateway"
    rows = [json.loads(line) for line in gateway.read_text().splitlines()]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[-1])
    elif mutation == "timeout":
        rows[-1]["event"] = "transport_error"
    else:
        rows[-1]["http_status"] = 200
    gateway.write_text("\n".join(map(json.dumps, rows)))
    evidence = tmp_path / "proof"
    proof = json.loads(evidence.read_text())
    proof["gateway_sha256"] = hashlib.sha256(gateway.read_bytes()).hexdigest()
    evidence.write_text(json.dumps(proof))
    with pytest.raises(ReplayRefused):
        SettledSequence(tmp_path / "saved", evidence, gateway, tmp_path)
