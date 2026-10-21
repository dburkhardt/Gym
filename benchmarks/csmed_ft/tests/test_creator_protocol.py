# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from benchmarks.csmed_ft.creator_protocol import SOURCE_SHA256, ChunkPrediction, merge_chunks, parse_label


def creator_function(filename, name):
    root = Path(__file__).resolve().parents[4] / "csmed-baselines-upstream/experiments/csmed_ft"
    path = root / filename
    if not path.exists():
        pytest.skip("Pinned creator checkout is not installed")
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256[filename]
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    scope = {"copy": copy}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


@pytest.mark.parametrize(
    "answer", ["Included", "INCLUDE", "yes", "Excluded", "EXCLUDE", "no", "", " Included", "yes\n", "unknown"]
)
def test_parser_exact_creator(answer):
    parser = creator_function("full_text_prompting.py", "parse_response")
    response = {"choices": [SimpleNamespace(message=SimpleNamespace(content=answer))]}
    expected = parser(response, {"review_id": "r", "pmid": "p", "label": 1}, 0, 0)
    assert parse_label(answer) == expected["prediction"]


def test_merge_exact_creator_and_exposes_invalid_reference():
    chunks = [
        ChunkPrediction(2, -1, 0),
        ChunkPrediction(1, 1, 1),
        ChunkPrediction(2, -1, 1),
        ChunkPrediction(1, 0, 1),
        ChunkPrediction(3, -1, 1),
        ChunkPrediction(3, 1, 1),
    ]
    original = creator_function("analyse_results.py", "merge_predictions")
    expected = original(pd.DataFrame([vars(chunk) for chunk in chunks])).to_dict("records")
    actual = merge_chunks(chunks)
    assert [{key: getattr(row, key) for key in ("uuid", "prediction", "gold_label")} for row in actual] == expected
    assert [row.prediction for row in actual] == [0, 1, 1]
    assert actual[1].invalid_chunk_count == 2
    assert actual[1].conflicting_gold_labels is True
    assert actual[1].gold_label == 0  # Preserve creator first-gold behavior, not majority vote.


def test_contract_does_not_accept_unbound_or_invalid_records():
    assert merge_chunks([]) == []
    with pytest.raises(ValueError, match="integer row"):
        merge_chunks([ChunkPrediction("1", 1, 0)])
    with pytest.raises(ValueError, match="prediction"):
        merge_chunks([ChunkPrediction(1, 2, 0)])


def test_grouped_complete_missing_and_mixed_scope():
    from benchmarks.csmed_ft.creator_protocol import PROTOCOL
    from benchmarks.csmed_ft.grouped_report import report

    inputs, outputs = [], []
    for paper, gold, answers in [(1, "included", ["yes", "unknown"]), (2, "excluded", ["no", "yes"])]:
        for chunk, answer in enumerate(answers):
            key = f"{paper}:{chunk}"
            meta = dict(
                paper_id=str(paper),
                review_id="r",
                uuid=paper,
                chunk_count=2,
                paper_chunk_id=chunk,
                answer=gold,
                protocol=PROTOCOL,
                recipe_sha256="bound",
                generation_protocol={"model": "fixture", "temperature": 0.3, "max_output_tokens": 10},
            )
            request = {"input": [{"role": "user", "content": key}], "temperature": 0.3, "max_output_tokens": 10}
            inputs.append(dict(task_id=key, verifier_metadata=meta, responses_create_params=request))
            outputs.append(
                dict(
                    task_id=key,
                    verifier_metadata=meta,
                    responses_create_params=request,
                    response={
                        "model": "fixture",
                        "output": [{"type": "message", "content": [{"type": "output_text", "text": answer}]}],
                    },
                )
            )
    actual = report(inputs, outputs)
    creator = creator_function("analyse_results.py", "get_evaluation")
    # Inject only explicit sklearn symbols required by the extracted pure function.
    from sklearn import metrics

    creator.__globals__.update(
        {
            name: getattr(metrics, name)
            for name in ["accuracy_score", "precision_score", "recall_score", "f1_score", "fbeta_score"]
        }
    )
    assert actual["metrics"] == creator([1, 0], [1, 0])
    assert actual["metric_denominator"] == 2
    altered = copy.deepcopy(outputs)
    altered[0]["responses_create_params"]["input"][0]["content"] = "altered"
    with pytest.raises(ValueError, match="prompt"):
        report(inputs, altered)
    wrong_model = copy.deepcopy(outputs)
    wrong_model[0]["response"]["model"] = "another-model"
    with pytest.raises(ValueError, match="model/settings"):
        report(inputs, wrong_model)
    missing = report(inputs, outputs[:-1])
    assert missing["unresolved_papers"] == 1 and missing["metric_denominator"] == 1
    masked = copy.deepcopy(outputs)
    masked[0]["mask_sample"] = True
    assert report(inputs, masked)["unresolved_papers"] == 1
    failed = copy.deepcopy(outputs)
    failed[0]["response"]["status"] = "failed"
    assert report(inputs, failed)["unresolved_papers"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        report(inputs, outputs + outputs[:1])
    changed = copy.deepcopy(inputs)
    changed[0]["verifier_metadata"]["recipe_sha256"] = "different"
    with pytest.raises(ValueError, match="Mixed"):
        report(changed, outputs)


def test_recipe_unknown_refused_and_exact_prompt_construction():
    from benchmarks.csmed_ft.chunk_preparation import QUESTION, chunk_paper, load_recipe

    with pytest.raises(ValueError, match="Unknown"):
        load_recipe({"recipe_model": "nemotron-unspecified"})
    calls = []

    class Splitter:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def split_text(self, text):
            return [text, "second"]

    encoding = SimpleNamespace(encode=lambda text: list(range(9000)))
    prompts = chunk_paper(
        dict(review_title="R", review_criteria="C", title="T", abstract="A", main_text="M"),
        recipe_model="gpt-3.5-turbo-16k",
        encoding=encoding,
        splitter=Splitter,
    )
    assert calls[0]["chunk_size"] == 8000 and calls[1]["chunk_size"] == 8000
    assert prompts[0] == QUESTION + "\n\nSystematic review:\nR\nC\n\nCandidate paper:\nT\n\nA\n\nM\n\nAnswer: "
    assert prompts[1].endswith("\n\nsecond\n\nAnswer: ")
