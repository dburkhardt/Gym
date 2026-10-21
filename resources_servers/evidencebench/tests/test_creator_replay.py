# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

import pytest

from resources_servers.evidencebench.creator_replay import replay


ROOT = Path(os.environ.get("EVIDENCEBENCH_SOURCE", "../evidencebench-upstream"))
pytestmark = pytest.mark.skipif(not (ROOT / ".git").exists(), reason="Pinned creator checkout required")


def row(answer, key="one"):
    return {
        "task_id": key,
        "prompt": "unused",
        "answer": answer,
        "verifier_metadata": {"sentence_index2aspects": {"0": ["a"], "1": ["b"], "2": []}, "limit": 2},
    }


def test_actual_creator_loop():
    rows = [row("DECISION: [0, 1]"), row("DECISION: [0, 1, 2]", "two"), row("bad", "three")]
    a = replay(rows, creator_root=str(ROOT), parser="parse_gpt_output")
    assert a == replay(rows, creator_root=str(ROOT), parser="parse_gpt_output")
    assert a["scores"][0] == 1 and a["scores"][2] == 0
    assert abs(a["scores"][1] - 2 / 3) < 0.005
    assert a["rng_before"] != a["rng_after"]


def test_creator_errors_not_zero():
    with pytest.raises(ValueError):
        replay([row("DECISION: [0, ]")], creator_root=str(ROOT), parser="parse_gpt_output")
    with pytest.raises(KeyError):
        replay([row("DECISION: [0, 1, 99]")], creator_root=str(ROOT), parser="parse_gpt_output")


def test_explicit_mapping():
    with pytest.raises(ValueError, match="Explicit"):
        replay([], creator_root=str(ROOT), parser="nemotron")
    with pytest.raises(ValueError, match="Duplicate"):
        replay([row(""), row("")], creator_root=str(ROOT), parser="parse_gpt_output")


def test_no_rng_under_limit():
    a = replay([row("DECISION: [0]")], creator_root=str(ROOT), parser="parse_gpt_output")
    assert a["scores"] == [0.5] and a["rng_before"] == a["rng_after"]


def test_prompt_is_unused_by_pinned_parser():
    first = row("DECISION: [0]")
    second = dict(first, prompt="completely different scientific prompt")
    assert (
        replay([first], creator_root=str(ROOT), parser="parse_gpt_output")["scores"]
        == replay([second], creator_root=str(ROOT), parser="parse_gpt_output")["scores"]
    )


def test_cli_failed_records_preserve_null_and_identity(tmp_path, monkeypatch):
    import json

    from resources_servers.evidencebench.creator_replay import main

    valid = {
        "verifier_metadata": row("")["verifier_metadata"] | {"task_id": "one"},
        "response": {"output": [{"type": "message", "content": [{"type": "output_text", "text": "DECISION: [0]"}]}]},
        "reward": None,
        "mask_sample": True,
        "failure_kind": "verifier_error",
        "_ng_task_index": 4,
    }
    missing = {
        "task_id": "two",
        "reward": None,
        "mask_sample": True,
        "failure_kind": "provider_unknown",
        "_ng_task_index": 5,
    }
    source = tmp_path / "in.jsonl"
    dest = tmp_path / "out.json"
    source.write_text("\n".join(json.dumps(x) for x in [valid, missing]))
    monkeypatch.setattr(
        "sys.argv",
        [
            "replay",
            "--input",
            str(source),
            "--output",
            str(dest),
            "--creator-root",
            str(ROOT),
            "--parser",
            "parse_gpt_output",
        ],
    )
    main()
    result = json.loads(dest.read_text())
    assert result["record_scores"] == [0.5, None]
    assert result["deltas"] == [None, None]
    assert result["original_masks"] == [True, True]
    assert result["records"][1]["identity"]["_ng_task_index"] == 5
    assert result["records"][1]["projection_disposition"] == "unscored_missing_response"
