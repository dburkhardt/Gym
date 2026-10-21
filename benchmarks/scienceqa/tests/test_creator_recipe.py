# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
import json
import random
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.scienceqa.creator_recipe import PROTOCOL, prepare, score_ordered
from benchmarks.scientific_agent_campaign.deterministic import module


REPO = Path(__file__).resolve().parents[4] / "scienceqa-upstream"
pytestmark = pytest.mark.skipif(not REPO.exists(), reason="Pinned ScienceQA creator checkout required")
IDS = ["17752", "11434", "3634", "12382", "7598"]


def test_exact_creator_prompt_and_inference_contract():
    packet = prepare(REPO, IDS)
    problems = json.loads((REPO / "data/scienceqa/problems.json").read_text())
    creator = module(REPO / "models/base_prompt.py")
    args = SimpleNamespace(prompt_format="QCM-ALE", use_caption=False, options=list("ABCDE"))
    assert packet["shot_ids"] == ["6493", "16241"]
    for row in packet["solver_requests"]:
        assert row["protocol"] == PROTOCOL
        assert row["request"]["prompt"] == creator.build_prompt(problems, packet["shot_ids"], row["task_id"], args)
        assert row["request"]["prompt"].endswith("Answer:")
        assert set(row) == {"task_id", "protocol", "request"}
        assert "answer" not in row["request"] and "verifier_metadata" not in row["request"]
        assert row["request"]["stop"] == ["\n"] and row["request"]["max_tokens"] == 512
        assert row["request"]["temperature"] == 0
    # No caption is supplied to the creator builder; 11434's baseball caption is not introduced.
    assert "baseball" not in packet["solver_requests"][1]["request"]["prompt"].lower()


def creator_scores(outputs, packet):
    nodes = ast.parse((REPO / "models/run_gpt3.py").read_text()).body
    funcs = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in {"get_gpt3_result", "get_pred_idx"}]
    rng = random.Random(3)
    splits = json.loads((REPO / "data/scienceqa/pid_splits.json").read_text())
    assert rng.sample(splits["train"], 2) == packet["shot_ids"]
    answers = iter(outputs)
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return {"choices": [{"text": next(answers)}]}

    namespace = {"random": rng, "re": re, "openai": SimpleNamespace(Completion=SimpleNamespace(create=create))}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(REPO / "models/run_gpt3.py"), "exec"), namespace)
    args = SimpleNamespace(
        engine="text-davinci-002",
        temperature=0.0,
        max_tokens=512,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    indexes = []
    for row, reference in zip(packet["solver_requests"], packet["evaluator_references"]):
        answer, _ = namespace["get_gpt3_result"](row["request"]["prompt"], args)
        indexes.append(namespace["get_pred_idx"](answer, reference["choices"], list("ABCDE")))
        assert {k: v for k, v in requests[-1].items() if k != "engine"} == row["request"]
    return indexes


@pytest.mark.parametrize("kind", ["correct", "wrong", "empty", "malformed", "duplicate"])
def test_scoring_and_raw_request_match_actual_creator_functions(kind):
    packet = prepare(REPO, IDS)
    texts = []
    for ref in packet["evaluator_references"]:
        letter = "ABCDE"[ref["answer"]]
        texts.append(
            {
                "correct": f"The answer is {letter}.",
                "wrong": f"The answer is {'ABCDE'[(ref['answer'] + 1) % len(ref['choices'])]}.",
                "empty": "",
                "malformed": f"({letter})",
                "duplicate": f"The answer is {letter}. The answer is {letter}.",
            }[kind]
        )
    actual = score_ordered(REPO, packet, texts)
    assert [x["predicted_index"] for x in actual] == creator_scores(texts, packet)
    if kind == "correct":
        assert all(x["strict_accuracy"] == 1 for x in actual)
    elif kind == "wrong":
        assert all(x["strict_accuracy"] == 0 and not x["fallback_randomized"] for x in actual)
    else:
        assert all(x["fallback_randomized"] and x["measurement_valid"] is False for x in actual)


def test_reject_duplicate_ids_and_modified_packet():
    with pytest.raises(ValueError, match="unique"):
        prepare(REPO, [IDS[0], IDS[0]])
    packet = prepare(REPO, IDS)
    packet["solver_requests"][0]["request"]["max_tokens"] = 16384
    with pytest.raises(ValueError, match="differs"):
        score_ordered(REPO, packet, [""] * 5)


def test_native_preparation_and_ordered_saved_regrade(tmp_path, monkeypatch):
    from benchmarks.scienceqa import prepare_recipe
    from benchmarks.scienceqa.regrade_recipe import regrade

    sample = tmp_path / "frozen.jsonl"
    sample.write_text("\n".join(json.dumps({"uuid": key}) for key in IDS) + "\n")
    monkeypatch.setenv("CAMPAIGN_SAMPLE_JSONL", str(sample))
    monkeypatch.setenv("SCIENCEQA_CREATOR_REPO", str(REPO))
    monkeypatch.setattr(prepare_recipe, "__file__", str(tmp_path / "prepare_recipe.py"))
    staged = prepare_recipe.prepare()
    rows = [json.loads(line) for line in staged.read_text().splitlines()]
    packet = prepare(REPO, IDS)
    assert [row["task_id"] for row in rows] == IDS
    for row, expected in zip(rows, packet["solver_requests"]):
        assert json.loads(row["responses_create_params"]["input"][0]["content"]) == expected["request"]
    outputs = ["invalid", "The answer is A.", "", "The answer is A.", "invalid"]
    result_rows = []
    for key, text in zip(IDS, outputs):
        result_rows.append(
            {
                "task_id": key,
                "response": {
                    "id": key,
                    "created_at": 1,
                    "model": "fixture",
                    "object": "response",
                    "status": "completed",
                    "parallel_tool_calls": False,
                    "tool_choice": "none",
                    "tools": [],
                    "output": [
                        {
                            "id": key,
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": text, "annotations": []}],
                        }
                    ],
                },
            }
        )
    saved = tmp_path / "saved.jsonl"
    saved.write_text("\n".join(json.dumps(row) for row in reversed(result_rows)) + "\n")
    result = regrade(REPO, staged.with_suffix(".packet.json"), saved)
    assert result["results"] == score_ordered(REPO, packet, outputs)
    assert result["model_calls"] == 0
    saved.write_text(json.dumps(result_rows[0]) + "\n")
    with pytest.raises(ValueError, match="exactly"):
        regrade(REPO, staged.with_suffix(".packet.json"), saved)
