import hashlib
import json
import socket
from pathlib import Path

import pytest

from benchmarks.metasyn.import_saved import Inputs, reconcile


ROOT = Path(__file__).resolve().parents[5]
ARTIFACTS = ROOT / "outputs/broad-eval-campaign/harbor/agent-expansion"
MATERIALIZED = ARTIFACTS / "metasyn-materialized"
SOURCE = ARTIFACTS / "sources/111/THUIR-MetaSyn-3ee040f/src/metasyn"
TRANSPORT = ARTIFACTS / "metasyn-transport"


def run():
    if not MATERIALIZED.exists():
        pytest.skip("Preserved campaign artifacts unavailable")
    return reconcile(MATERIALIZED, SOURCE, TRANSPORT)


def test_actual_saved20_parity_without_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("Network is forbidden for saved import")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    rows, proof = run()
    for path, expected in proof["source_hashes"].items():
        source = Path(path)
        assert source.is_file(), path
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, path
    assert len(rows) == 20
    assert len({b["review_id"] for b in proof["bindings"]}) == 20
    assert proof["new_provider_calls"] == 0
    assert proof["provider_evidence"]["remaining15_raw_pairs_verified"] == 75
    assert len(proof["provider_evidence"]["remaining15_generators_bound"]) == 15
    assert proof["provider_evidence"]["pilot5_raw_bodies"] == "unavailable"
    assert proof["provider_evidence"]["unknown_historical_requests"].startswith("unknown:")
    summary = proof["combined_metric_summary"]
    assert summary["per_metric"]["inclusion_f1"]["reviewed_eligible_scores"] == 20
    assert summary["per_metric"]["inclusion_f1"]["reviewed_denominator"] == 20
    assert sorted(
        group["denominator"] for group in summary["per_metric"]["inclusion_f1"]["reviewed_source_groups"]
    ) == [5, 15]
    assert summary["per_metric"]["structure_quality"]["unresolved"] == 20
    originals = []
    for file in ("gym-pilot5-results.jsonl", "gym-remaining15-results.jsonl"):
        originals.extend(json.loads(line) for line in (MATERIALIZED / file).read_text().splitlines())
    assert [{k: v for k, v in row.items() if k != "metadata"} for row in rows] == originals


@pytest.mark.parametrize(
    "target,mutation,error",
    [
        ("gym-pilot5-results.jsonl", "prompt", "prompt mismatch"),
        ("pilot5/203/evaluation.json", "score", "score differs"),
        ("pilot5/203/evaluation.json", "protocol", "protocol mismatch"),
        ("remaining15-raw-body-audit.json", "hash", "checksum mismatch"),
    ],
)
def test_historical_tampering_refused(monkeypatch, target, mutation, error):
    original = Inputs.read

    def altered(self, path):
        raw = original(self, path)
        if str(path).endswith(target):
            if mutation == "prompt":
                rows = [json.loads(line) for line in raw.splitlines()]
                rows[0]["responses_create_params"]["input"][0]["content"] = "different"
                return "\n".join(json.dumps(r) for r in rows).encode()
            value = json.loads(raw)
            if mutation == "score":
                value["metrics"]["inclusion_f1"] = -1
            elif mutation == "protocol":
                value["evaluator"]["selection_source"] = "json"
            else:
                value["artifacts"][0]["request_sha256"] = "0" * 64
            return json.dumps(value).encode()
        return raw

    monkeypatch.setattr(Inputs, "read", altered)
    with pytest.raises(ValueError, match=error):
        run()


def test_symlink_and_missing_inputs_refused(tmp_path):
    source = tmp_path / "source"
    source.write_text("{}")
    alias = tmp_path / "alias"
    alias.symlink_to(source)
    for path in (alias, tmp_path / "missing"):
        with pytest.raises(ValueError, match="regular nonsymlink"):
            Inputs().read(path)


def test_changed_creator_refused(monkeypatch):
    original = Inputs.read

    def altered(self, path):
        data = original(self, path)
        return data + b"\n" if str(path).endswith("/evaluation.py") else data

    monkeypatch.setattr(Inputs, "read", altered)
    with pytest.raises(ValueError, match="Creator source checksum"):
        run()


def test_cli_refuses_existing_output(monkeypatch, tmp_path):
    from benchmarks.metasyn import import_saved

    output = tmp_path / "existing"
    output.mkdir()
    original = output / "imported-results.jsonl"
    original.write_text("preserved")
    monkeypatch.setattr(import_saved, "reconcile", lambda *args: ([], {}))
    monkeypatch.setattr(
        "sys.argv",
        ["import_saved", "--materialized", ".", "--creator-source", ".", "--transport", ".", "--output", str(output)],
    )
    with pytest.raises(FileExistsError):
        import_saved.main()
    assert original.read_text() == "preserved"
