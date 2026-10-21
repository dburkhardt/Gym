# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "mixed_control", Path(__file__).parents[1] / "e2e/coordinator/mixed_scientific_restart_control.py"
)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def test_exact_transcript_rejection_then_saved_attempt():
    mol = {"input": "molecular task", "model": control.MODEL}
    captures = [{"request": {"input": f"scientific step {i}"}, "response": {"output": i}} for i in range(2)]
    bodies = [
        mol,
        dict(captures[0]["request"], model=control.MODEL),
        dict(captures[0]["request"], model=control.MODEL),
        dict(captures[1]["request"], model=control.MODEL),
    ]
    replies = [control.fixture_reply(i, body, mol, {"output": "saved"}, captures) for i, body in enumerate(bodies, 1)]
    assert [status for status, _ in replies] == [200, 429, 200, 200]
    assert replies[2][1] is captures[0]["response"]
    assert replies[3][1] is captures[1]["response"]
    with pytest.raises(ValueError, match="extra"):
        control.fixture_reply(5, bodies[-1], mol, {}, captures)
    with pytest.raises(ValueError, match="mismatch"):
        control.fixture_reply(3, bodies[-1], mol, {}, captures)
    with pytest.raises(ValueError, match="mismatch"):
        control.fixture_reply(1, dict(mol, temperature=0), mol, {}, captures)


def test_all_invocations_share_absolute_deadline(monkeypatch):
    monkeypatch.setattr(control.time, "monotonic", lambda: 10)
    assert control.remaining(20) == 10
    monkeypatch.setattr(control.time, "monotonic", lambda: 19)
    assert control.remaining(20) == 1
    monkeypatch.setattr(control.time, "monotonic", lambda: 20)
    with pytest.raises(TimeoutError):
        control.remaining(20)


@pytest.mark.parametrize("message", ["Error: No such object: exact-owned", "error: no such object: exact-owned"])
def test_container_absence_case_variants(message):
    from types import SimpleNamespace

    control.require_container_absence("exact-owned", SimpleNamespace(returncode=1, stderr=message))


@pytest.mark.parametrize("code,message", [(0, ""), (1, "permission denied"), (1, "No such object: unrelated")])
def test_container_absence_refuses_unproven(code, message):
    from types import SimpleNamespace

    with pytest.raises(RuntimeError):
        control.require_container_absence("exact-owned", SimpleNamespace(returncode=code, stderr=message))


@pytest.fixture
def planned_rows():
    return [
        {"task_source": "moljson_resources", "uuid": "mol-id", "task_id": "mol-id", "_ng_task_index": 0},
        {"task_source": "scicode_benchmark_agent", "uuid": "20", "task_id": "20", "_ng_task_index": 1},
    ]


def test_reads_plan_only_artifact_not_execution_file(tmp_path, planned_rows):
    output = tmp_path / "results.jsonl"
    (tmp_path / "results_planned_inputs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in planned_rows))
    (tmp_path / "results_materialized_inputs.jsonl").write_text("invalid execution artifact")
    assert control.read_planned_inputs(output, "mol-id") == planned_rows


@pytest.mark.parametrize("mutation", ["reversed", "missing", "identity", "indices", "nonobject", "malformed"])
def test_plan_rejects_malformed_or_crosswired(tmp_path, planned_rows, mutation):
    rows = planned_rows
    if mutation == "reversed":
        rows.reverse()
    elif mutation == "missing":
        rows.pop()
    elif mutation == "identity":
        rows[0]["task_id"] = "other"
    elif mutation == "indices":
        rows[1]["_ng_task_index"] = 0
    elif mutation == "nonobject":
        rows[1] = []
    content = "{" if mutation == "malformed" else "".join(json.dumps(row) + "\n" for row in rows)
    (tmp_path / "results_planned_inputs.jsonl").write_text(content)
    with pytest.raises(ValueError):
        control.read_planned_inputs(tmp_path / "results.jsonl", "mol-id")
