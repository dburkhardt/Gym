# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from benchmarks.moosechem.remaining_fifteen import ensure_unattempted, tasks_from_sample


IDS = [18, 26, 38, 0, 33, 24, 39, 44, 28, 36, 22, 11, 50, 10, 12, 25, 19, 35, 14, 8]


def test_frozen_disjoint_population(tmp_path):
    p = tmp_path / "sample.jsonl"
    p.write_text("\n".join(json.dumps({"id": i}) for i in IDS))
    assert tasks_from_sample(p) == IDS[5:]
    assert not set(tasks_from_sample(p)) & {18, 26, 38, 0, 33}
    p.write_text(p.read_text().replace('"id": 24', '"id": 26'))
    with pytest.raises(AssertionError):
        tasks_from_sample(p)


@pytest.mark.parametrize(
    "receipt",
    [
        "run-intent.json",
        "dispatch-intents.jsonl",
        "model-traces.jsonl",
        "queue-dispatch-intent.json",
        "child-identity.json",
        "child-release.json",
        "parent-child-receipt.json",
        "result.json",
        "queue-terminal.json",
    ],
)
def test_no_retry_even_without_terminal_response(tmp_path, receipt):
    ensure_unattempted(tmp_path, IDS[5:])
    d = tmp_path / "task24-interrupted"
    d.mkdir()
    (d / receipt).write_text("")
    with pytest.raises(AssertionError):
        ensure_unattempted(tmp_path, IDS[5:])


def test_prior_calibration_is_not_relaunched(tmp_path):
    d = tmp_path / "task26-uncertain"
    d.mkdir()
    (d / "dispatch-intents.jsonl").write_text("")
    ensure_unattempted(tmp_path, IDS[5:])
