import gzip
import json

import pytest

from benchmarks.agentideabench.release import sha256, validate_release


def fixture_release(tmp_path):
    data = tmp_path / "table.csv.gz"
    with gzip.open(data, "wt") as stream:
        stream.write('id,text\n1,"text, with comma"\n')
    manifest = {
        "release": "AgentIdeaBench",
        "dropped_columns": ["abstract"],
        "redaction": "fixture",
        "files": [{"path": data.name, "bytes": data.stat().st_size, "sha256": sha256(data),
                   "rows": 1, "columns": ["id", "text"]}],
    }
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps(manifest))
    return path, data, manifest


def test_valid_release_stays_incomplete(tmp_path):
    path, _, _ = fixture_release(tmp_path)
    result = validate_release(tmp_path, expected_manifest_sha256=sha256(path))
    assert result["status"] == "released_tables_verified_not_complete_benchmark_inputs"
    assert result["files"][0]["rows"] == 1
    assert result["dropped_columns"] == ["abstract"]


def test_manifest_and_table_tamper(tmp_path):
    path, data, _ = fixture_release(tmp_path)
    with pytest.raises(ValueError, match="binding"):
        validate_release(tmp_path, expected_manifest_sha256="0" * 64)
    data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        validate_release(tmp_path, expected_manifest_sha256=sha256(path))


@pytest.mark.parametrize("change,match", [
    ({"path": "../escape.csv.gz"}, "Unsafe"),
    ({"rows": 2}, "row count"),
    ({"columns": ["wrong", "header"]}, "schema"),
])
def test_bound_manifest_semantics(tmp_path, change, match):
    path, _, manifest = fixture_release(tmp_path)
    manifest["files"][0].update(change)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        validate_release(tmp_path, expected_manifest_sha256=sha256(path))
