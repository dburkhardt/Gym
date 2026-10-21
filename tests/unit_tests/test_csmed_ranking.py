import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.csmed.prepare import prepare
from resources_servers.csmed_ranking.grading import read_json, validate_rankings


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "csmed-baselines-upstream/experiments/csmed_cochrane"


def fixture():
    qrels = {f"q{i}": {"a": 1, "b": 1, "c": 0, "d": 0} for i in range(5)}
    good = {q: [["a", 4], ["b", 3], ["c", 2], ["d", 1]] for q in qrels}
    bad = {q: list(reversed([["a", 1], ["b", 2], ["c", 3], ["d", 4]])) for q in qrels}
    tied = {q: [["d", 1], ["b", 1], ["a", 1], ["c", 1]] for q in qrels}
    return qrels, {"oracle": good, "reversed": bad, "tied": tied}


def test_actual_creator_full_report_parity(tmp_path):
    import os
    import subprocess

    interpreter = os.environ.get("CSMED_TEST_PYTHON")
    if not interpreter:
        pytest.skip("Set CSMED_TEST_PYTHON to qualified ranx0.3.18 interpreter")
    qrels, runs = fixture()
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps({"qrels": qrels, "runs": runs}))
    script = """
import json,sys
from pathlib import Path
from resources_servers.csmed_ranking.grading import creator,read_json,score_runs,validate_rankings
source,fixture,target=map(Path,sys.argv[1:])
x=read_json(fixture);qrels=x['qrels'];runs=x['runs']
result=score_runs(source,runs,qrels,target/'maintained')
direct=creator(source);(target/'direct').mkdir()
direct.evaluate_runs({n:validate_rankings(r,qrels) for n,r in runs.items()},qrels,outfile_path=str(target/'direct'))
assert (target/'direct/report.tex').read_bytes()==(target/'maintained/report.tex').read_bytes()
assert read_json(target/'direct/other_measures.json')==result['supplementary']
assert len(result['metric_names'])==21
assert result['metrics']['oracle']['ndcg']==1
assert result['metrics']['reversed']['map']<1
assert result['review_count']==5 and result['scalar_reward'] is None
"""
    result = subprocess.run(
        [interpreter, "-c", script, str(SOURCE / "measures.py"), str(fixture_path), str(tmp_path)],
        cwd=ROOT,
        env=dict(
            os.environ,
            PYTHONPATH=str(ROOT),
            IR_DATASETS_HOME=str(tmp_path / "ir"),
            MPLCONFIGDIR=str(tmp_path / "mpl"),
            NUMBA_CACHE_DIR=str(tmp_path / "numba"),
        ),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "kind", ["missing_review", "missing_doc", "duplicate", "unknown", "nan", "ascending", "boolean", "degenerate"]
)
def test_invalid_population_or_scores_refuse(kind):
    qrels, runs = fixture()
    run = runs["oracle"]
    if kind == "missing_review":
        del run["q0"]
    elif kind == "missing_doc":
        run["q0"].pop()
    elif kind == "duplicate":
        run["q0"][1][0] = "a"
    elif kind == "unknown":
        run["q0"][1][0] = "x"
    elif kind == "nan":
        run["q0"][0][1] = float("nan")
    elif kind == "ascending":
        run["q0"].reverse()
    elif kind == "boolean":
        run["q0"][0][1] = True
    else:
        qrels["q0"] = dict.fromkeys(qrels["q0"], 0)
    with pytest.raises(ValueError):
        validate_rankings(run, qrels)


def test_ties_preserve_order_and_duplicate_json_refuses(tmp_path):
    qrels, runs = fixture()
    for pairs in runs["oracle"].values():
        for pair in pairs:
            pair[1] = 1
    assert list(validate_rankings(runs["oracle"], qrels)["q0"]) == ["a", "b", "c", "d"]
    path = tmp_path / "duplicate.json"
    path.write_text('{"q": {}, "q": {}}')
    with pytest.raises(ValueError, match="Duplicate"):
        read_json(path)


def test_prepare_exact_creator_fields_qrels_isolated(tmp_path):
    body = {
        "dataset_details": {
            "title": "title",
            "abstract": "abstract",
            "criteria": {"include": "criteria"},
            "search_strategy": {"pubmed": "query"},
        },
        "data": {
            "train": [
                {"document_id": "a", "text": "document A", "labels": [1]},
                {"document_id": "b", "text": "document B", "labels": [0]},
            ]
        },
    }
    data = tmp_path / "eval.json"
    data.write_text(json.dumps({"EVAL": {"review": body}}))
    sha = hashlib.sha256(data.read_bytes()).hexdigest()
    output = prepare(SOURCE / "csmed_cochrane_retrieval.py", data, sha, tmp_path / "prepared")
    inputs = read_json(output / "review-inputs.json")
    assert inputs["review"]["queries"] == {
        "title": "title",
        "abstract": "abstract",
        "criteria": "include: criteria",
        "query": "query",
    }
    assert inputs["review"]["collection"] == [{"id": "a", "text": "document A"}, {"id": "b", "text": "document B"}]
    assert read_json(output / "qrels.json") == {"review": {"a": 1, "b": 0}}
    with pytest.raises(FileExistsError):
        prepare(SOURCE / "csmed_cochrane_retrieval.py", data, sha, output)
    data.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        prepare(SOURCE / "csmed_cochrane_retrieval.py", data, sha, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("kind", ["missing_text", "empty_text", "missing_labels", "empty_labels", "bool_label"])
def test_preparation_incomplete_documents_refuse_before_publication(tmp_path, kind):
    docs = [{"document_id": "a", "text": "A", "labels": [1]}, {"document_id": "b", "text": "B", "labels": [0]}]
    if kind == "missing_text":
        del docs[0]["text"]
    elif kind == "empty_text":
        docs[0]["text"] = ""
    elif kind == "missing_labels":
        del docs[0]["labels"]
    elif kind == "empty_labels":
        docs[0]["labels"] = []
    else:
        docs[0]["labels"] = [True]
    body = {
        "dataset_details": {
            "title": "title",
            "abstract": "abstract",
            "criteria": {"include": "criteria"},
            "search_strategy": {"pubmed": "query"},
        },
        "data": {"train": docs},
    }
    data = tmp_path / "eval.json"
    data.write_text(json.dumps({"EVAL": {"review": body}}))
    with pytest.raises(ValueError):
        prepare(
            SOURCE / "csmed_cochrane_retrieval.py",
            data,
            hashlib.sha256(data.read_bytes()).hexdigest(),
            tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()
