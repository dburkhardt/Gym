# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from benchmarks.moosechem.guarded import GuardedTransport, SettledContinuation, continuation_binding
from benchmarks.moosechem.native import TransportAbort
from benchmarks.moosechem.qualify_replay import ReplayRefused
from benchmarks.moosechem.test_qualify_settled_tail import sequence


def test_only_exact_tail_enters_guarded_transport(tmp_path, monkeypatch):
    seq, rows = sequence(tmp_path, monkeypatch)
    digest = continuation_binding(seq, tmp_path / "proof", "a" * 64)[1]
    calls = []

    def guarded(self, stage, trace, budget, payload):
        calls.append((stage, budget, payload))
        return "guarded"

    monkeypatch.setattr(GuardedTransport, "create", guarded)
    transport = SettledContinuation(
        "socket",
        "token",
        "new-lineage",
        digest,
        sequence=seq,
        output=tmp_path / "out",
        evidence=tmp_path / "proof",
        base_digest="a" * 64,
    )
    client = transport.factory(stage="generate", trace=tmp_path / "out/trace", output_token_budget=16384)
    assert client.chat.completions.create(**{**rows[0]["request"], "max_tokens": 8192}).id == "response"
    assert calls == []
    with pytest.raises(ReplayRefused):
        client.chat.completions.create(**{**rows[1]["request"], "max_tokens": 8192, "temperature": 0})
    assert calls == []
    assert client.chat.completions.create(**{**rows[1]["request"], "max_tokens": 8192}) == "guarded"
    assert calls == [("generate", 16384, {**rows[1]["request"], "max_tokens": 8192})]
    with pytest.raises(FileExistsError):
        SettledContinuation(
            "socket",
            "token",
            "new-lineage",
            digest,
            sequence=seq,
            output=tmp_path / "out",
            evidence=tmp_path / "proof",
            base_digest="a" * 64,
        )


def test_tail_uses_real_guarded_failure_latch(tmp_path, monkeypatch):
    seq, rows = sequence(tmp_path, monkeypatch)
    import httpx

    digest = continuation_binding(seq, tmp_path / "proof", "a" * 64)[1]
    calls = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["follow_redirects"] is False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, **kwargs):
            calls.append(kwargs)
            raise httpx.ConnectError("fixture refusal")

    monkeypatch.setattr(httpx, "Client", Client)
    t = SettledContinuation(
        "socket",
        "token",
        "lineage",
        digest,
        sequence=seq,
        output=tmp_path / "out",
        evidence=tmp_path / "proof",
        base_digest="a" * 64,
    )
    c = t.factory(stage="generate", trace=tmp_path / "out/model-traces.jsonl", output_token_budget=16384)
    c.chat.completions.create(**{**rows[0]["request"], "max_tokens": 8192})
    for _ in range(2):
        with pytest.raises(TransportAbort):
            c.chat.completions.create(**{**rows[1]["request"], "max_tokens": 8192})
    assert len(calls) == 1
    assert calls[0]["json"] == rows[1]["request"]
    assert calls[0]["headers"]["x-nemo-gym-logical-call-id"] == "moosechem:lineage:0"
    assert (tmp_path / "out/dispatch-intents.jsonl").exists()


def test_changed_binding_and_existing_attempt_refuse_before_writes(tmp_path, monkeypatch):
    seq, _ = sequence(tmp_path, monkeypatch)
    digest = continuation_binding(seq, tmp_path / "proof", "a" * 64)[1]
    with pytest.raises(ValueError, match="digest mismatch"):
        SettledContinuation(
            "socket",
            "token",
            "lineage",
            "b" * 64,
            sequence=seq,
            output=tmp_path / "bad",
            evidence=tmp_path / "proof",
            base_digest="a" * 64,
        )
    assert not (tmp_path / "bad").exists()
    out = tmp_path / "existing"
    out.mkdir()
    (out / "run-intent.json").write_text("preserve")
    with pytest.raises(ValueError, match="Existing attempt"):
        SettledContinuation(
            "socket",
            "token",
            "lineage",
            digest,
            sequence=seq,
            output=out,
            evidence=tmp_path / "proof",
            base_digest="a" * 64,
        )
    assert list(out.iterdir()) == [out / "run-intent.json"]
    assert (out / "run-intent.json").read_text() == "preserve"
